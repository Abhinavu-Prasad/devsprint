import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import folium
from folium import plugins
import timezonefinder
import pytz
from typing import Tuple, List, Dict
import warnings
import pickle
import calendar
import time
warnings.filterwarnings('ignore')

class LocationManager:
    """Manages location-related operations including geocoding and coordinate handling"""
    
    def __init__(self):
        """Initialize the location manager with a geocoder"""
        self.geolocator = Nominatim(user_agent="traffic_route_planner")
        # Create a rate-limited geocoding function to prevent hitting API limits
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1)
        
    def get_coordinates(self, address: str) -> Tuple[float, float]:
        """Convert address to coordinates"""
        try:
            location = self.geocode(address)
            if location is None:
                raise ValueError(f"Could not find coordinates for address: {address}")
            return (location.latitude, location.longitude)
        except Exception as e:
            raise ValueError(f"Error geocoding address '{address}': {str(e)}")
    
    def get_address(self, lat: float, lon: float) -> str:
        """Convert coordinates to address"""
        try:
            location = self.geolocator.reverse((lat, lon), exactly_one=True)
            if location is None:
                raise ValueError(f"Could not find address for coordinates ({lat}, {lon})")
            return location.address
        except Exception as e:
            raise ValueError(f"Error reverse geocoding coordinates ({lat}, {lon}): {str(e)}")

class MapManager:
    """Manages map data and operations"""
    
    def __init__(self, city: str = None):
        """Initialize the map manager"""
        self.city = city
        self.G = None
        self.load_map()
    
    def load_map(self):
        """Load or download map data"""
        try:
            if self.city:
                self.G = ox.graph_from_place(self.city, network_type='drive')
            else:
                # Default to a small area if no city is specified
                self.G = ox.graph_from_bbox(12.98, 12.88, 77.64, 77.54, 
                                          network_type='drive')
            
            # Add edge speeds and travel times
            self.G = ox.add_edge_speeds(self.G)
            self.G = ox.add_edge_travel_times(self.G)
            
        except Exception as e:
            print(f"Error loading map: {str(e)}")
            print("Loading default map...")
            self.G = ox.graph_from_bbox(12.98, 12.88, 77.64, 77.54, 
                                      network_type='drive')
            self.G = ox.add_edge_speeds(self.G)
            self.G = ox.add_edge_travel_times(self.G)
    
    def get_nearest_node(self, coords: Tuple[float, float]) -> int:
        """Get nearest node to coordinates"""
        return ox.distance.nearest_nodes(self.G, coords[1], coords[0])

class TrafficDataManager:
    """Manages traffic data and predictions"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.initialize_traffic_patterns()
    
    def initialize_traffic_patterns(self):
        """Initialize typical traffic patterns for different times"""
        hours = range(24)
        days = range(7)
        
        self.traffic_patterns = {}
        
        # Define peak hours and their multipliers
        peak_hours_morning = {7: 1.8, 8: 2.0, 9: 1.9, 10: 1.5}
        peak_hours_evening = {16: 1.6, 17: 1.9, 18: 2.0, 19: 1.8}
        
        for day in days:
            for hour in hours:
                key = (day, hour)
                multiplier = 1.0
                
                if hour in peak_hours_morning:
                    multiplier = peak_hours_morning[hour]
                elif hour in peak_hours_evening:
                    multiplier = peak_hours_evening[hour]
                
                if day in [5, 6]:  # Weekend adjustment
                    multiplier *= 0.7
                    
                self.traffic_patterns[key] = multiplier
    
    def get_traffic_multiplier(self, dt: datetime) -> float:
        """Get traffic multiplier for a specific datetime"""
        day = dt.weekday()
        hour = dt.hour
        return self.traffic_patterns.get((day, hour), 1.0)
    
    def simulate_real_time_traffic(self, base_multiplier: float) -> float:
        """Simulate real-time traffic conditions"""
        variation = np.random.normal(0, 0.1)
        return max(0.5, min(3.0, base_multiplier + variation))

class WeatherDataManager:
    """Manages weather data and its impact on traffic"""
    def __init__(self):
        self.weather_impacts = {
            'clear': 1.0,
            'cloudy': 1.1,
            'rain': 1.3,
            'heavy_rain': 1.5,
            'fog': 1.4
        }
    
    def simulate_weather_impact(self) -> float:
        """Simulate weather impact on traffic"""
        weather = np.random.choice(list(self.weather_impacts.keys()), 
                                 p=[0.5, 0.2, 0.15, 0.1, 0.05])
        return self.weather_impacts[weather]

class EnhancedRoutePlanner:
    def __init__(self, city: str = None):
        """Initialize the enhanced route planner"""
        self.city = city
        self.location_manager = LocationManager()
        self.map_manager = MapManager(city)
        self.traffic_manager = TrafficDataManager()
        self.weather_manager = WeatherDataManager()
        self.tf = timezonefinder.TimezoneFinder()
        
    def get_local_time(self, lat: float, lon: float) -> datetime:
        """Get local time for given coordinates"""
        timezone_str = self.tf.timezone_at(lat=lat, lng=lon)
        if timezone_str:
            local_tz = pytz.timezone(timezone_str)
            return datetime.now(local_tz)
        return datetime.now()
    
    def calculate_traffic_density(self, route: List[int], time: datetime) -> float:
        """Calculate traffic density for a route"""
        base_multiplier = self.traffic_manager.get_traffic_multiplier(time)
        real_time_multiplier = self.traffic_manager.simulate_real_time_traffic(base_multiplier)
        weather_multiplier = self.weather_manager.simulate_weather_impact()
        return base_multiplier * real_time_multiplier * weather_multiplier
    
    def estimate_travel_time(self, route: List[int], coords: List[Tuple[float, float]]) -> Dict:
        """Estimate travel time considering traffic and weather"""
        local_time = self.get_local_time(coords[0][0], coords[0][1])
        
        edges = ox.utils_graph.get_route_edge_attributes(self.map_manager.G, route)
        distance = sum(edge['length'] for edge in edges)
        base_speed = 40  # 40 km/h base speed
        base_time = (distance / 1000) / base_speed  # hours
        
        traffic_density = self.calculate_traffic_density(route, local_time)
        actual_time = base_time * traffic_density
        
        return {
            'base_time': base_time * 60,  # convert to minutes
            'actual_time': actual_time * 60,  # convert to minutes
            'distance': distance / 1000,  # convert to km
            'traffic_density': traffic_density,
            'local_time': local_time
        }
    
    def find_routes(self,source: str,destination: str, num_routes: int = 3) -> List[Dict]:
        """Find multiple routes with traffic information."""
        try:
            # Get coordinates for source and destination
            source_coords = self.location_manager.get_coordinates(source)
            dest_coords = self.location_manager.get_coordinates(destination)

            # Find the nearest nodes in the map
            source_node = self.map_manager.get_nearest_node(source_coords)
            dest_node = self.map_manager.get_nearest_node(dest_coords)

            routes = []
            route_infos = []

            for i in range(num_routes):
                try:
                    if i == 0:
                        # Shortest path
                        route = nx.shortest_path(self.map_manager.G, source_node, dest_node, weight='length')
                    else:
                        # Alternative paths with randomized weights
                        weights = nx.get_edge_attributes(self.map_manager.G, 'length')
                        for edge in weights:
                            weights[edge] *= np.random.uniform(1, 2)
                        route = nx.shortest_path(self.map_manager.G, source_node, dest_node, weight=weights.get)
                    
                    routes.append(route)
                except nx.NetworkXNoPath:
                    continue

            for i, route in enumerate(routes):
                coordinates = self.get_route_coordinates(route)
                estimates = self.estimate_travel_time(route, coordinates)

                route_infos.append({
                    'route_id': i + 1,
                    'coordinates': coordinates,
                    'base_time': estimates['base_time'],
                    'estimated_time': estimates['actual_time'],
                    'distance': estimates['distance'],
                    'traffic_density': estimates['traffic_density'],
                    'local_time': estimates['local_time']
                })

            return route_infos

        except Exception as e:
            print(f"Error finding routes: {str(e)}")
            return []

    
    def get_route_coordinates(self, route: List[int]) -> List[Tuple[float, float]]:
        """Convert route nodes to coordinates"""
        return [(self.map_manager.G.nodes[node]['y'], 
                self.map_manager.G.nodes[node]['x']) for node in route]
    
    def visualize_routes(self, routes: List[Dict]) -> folium.Map:
        """Create an enhanced folium map with traffic information"""
        first_route = routes[0]['coordinates']
        center_lat = sum(point[0] for point in first_route) / len(first_route)
        center_lon = sum(point[1] for point in first_route) / len(first_route)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        
        colormap = plugins.ColorMap(['green', 'yellow', 'orange', 'red'],
                                  vmin=1.0, vmax=2.0,
                                  caption='Traffic Density')
        colormap.add_to(m)
        
        for route_info in routes:
            traffic_density = route_info['traffic_density']
            
            if traffic_density < 1.3:
                color = 'green'
            elif traffic_density < 1.6:
                color = 'yellow'
            elif traffic_density < 1.9:
                color = 'orange'
            else:
                color = 'red'
            
            description = f"""
                Route {route_info['route_id']}:
                Estimated Time: {route_info['estimated_time']:.1f} mins
                Distance: {route_info['distance']:.2f} km
                Traffic Density: {traffic_density:.2f}x
                Local Time: {route_info['local_time'].strftime('%H:%M')}
            """
            
            folium.PolyLine(
                route_info['coordinates'],
                weight=4,
                color=color,
                opacity=0.8,
                popup=description
            ).add_to(m)
            
        return m

def main():
    print("Traffic-Aware Route Planner")
    print("---------------------------")

    city = "Bangalore, Karnataka, India"
    planner = EnhancedRoutePlanner(city)

    source = "Vidhana Soudha, Bengaluru, Karnataka 560001, India"
    destination = "Cubbon Park, Bengaluru, Karnataka 560001, India"

    print("\nFinding routes and analyzing traffic...")
    routes = planner.find_routes(source, destination)

    if routes:
        print("\nFound routes:")
        for route in routes:
            print(f"\nRoute {route['route_id']}:")
            print(f"Base travel time: {route['base_time']:.1f} minutes")
            print(f"Estimated time with traffic: {route['estimated_time']:.1f} minutes")
            print(f"Distance: {route['distance']:.2f} km")
            print(f"Traffic density: {route['traffic_density']:.2f}x normal")
            print(f"Local time: {route['local_time'].strftime('%Y-%m-%d %H:%M')}")

        print("\nGenerating traffic-aware map...")
        m = planner.visualize_routes(routes)

        map_file = "traffic_routes_map.html"
        m.save(map_file)
        print(f"\nMap saved to {map_file}")
        print("Note: Green = Light traffic, Yellow = Moderate, Orange = Heavy, Red = Very Heavy")
    else:
        print("No routes found.")
    
if __name__ == "__main__":
        main()
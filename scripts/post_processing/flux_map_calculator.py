import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os

@dataclass
class ReceiverGeometry:
    """Class to store receiver geometry parameters"""
    type: str  # "FLAT" or "CYLINDRICAL"
    # For flat receiver
    v1: Optional[np.ndarray] = None  # First vector defining the plane
    v2: Optional[np.ndarray] = None  # Second vector defining the plane
    anchor: Optional[np.ndarray] = None  # Origin point of the plane
    # For cylindrical receiver
    radius: Optional[float] = None
    height: Optional[float] = None
    center: Optional[np.ndarray] = None
    base_x: Optional[np.ndarray] = None  # Local x-axis direction
    base_z: Optional[np.ndarray] = None  # Local z-axis direction

@dataclass
class SunParameters:
    """Class to store sun parameters"""
    dni: float  # Direct Normal Irradiance (W/m²)
    sun_shape_area: float  # Area of sun shape in m²
    num_rays: int  # Number of rays used in simulation

class FluxMapCalculator:
    def __init__(self, 
                 receiver: ReceiverGeometry,
                 sun_params: SunParameters,
                 grid_size: Tuple[int, int] = (100, 100)):
        """
        Initialize the flux map calculator
        
        Args:
            receiver: ReceiverGeometry object containing receiver parameters
            sun_params: SunParameters object containing sun parameters
            grid_size: Tuple of (nx, ny) for the flux map resolution
        """
        self.receiver = receiver
        self.sun_params = sun_params
        self.nx, self.ny = grid_size
        self.flux_map = np.zeros(grid_size)
        
    def read_hit_points(self, filename: str, solver: str = "optix") -> np.ndarray:
        """
        Read hit points from CSV file
        
        Args:
            filename: Path to CSV file
            solver: "optix" or "solTrace"
            
        Returns:
            Array of hit points in global coordinates
        """
        hit_points = []
        
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            
            if solver == "optix":
                for row in reader:
                    number, stage = int(row[0]), int(row[1])
                    if stage == 2:  # Only keep receiver hits
                        x, y, z = map(float, row[2:5])
                        hit_points.append([x, y, z])
            else:  # solTrace
                for row in reader:
                    x, y, z = map(float, row[0:3])
                    element = row[6]
                    if element == "-1":  # Receiver hits
                        hit_points.append([x, y, z])
        
        return np.array(hit_points)
    
    def project_to_receiver(self, hit_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project hit points onto receiver surface
        
        Args:
            hit_points: Array of hit points in global coordinates
            
        Returns:
            Tuple of (x_local, y_local) coordinates on receiver surface
        """
        if self.receiver.type == "FLAT":
            return self._project_to_flat_receiver(hit_points)
        else:
            return self._project_to_cylindrical_receiver(hit_points)
    
    def _project_to_flat_receiver(self, hit_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project points onto flat receiver"""
        # Calculate receiver center and transformation matrix
        receiver_center = self.receiver.anchor + 0.5 * self.receiver.v1 + 0.5 * self.receiver.v2
        v1_norm = self.receiver.v1 / np.linalg.norm(self.receiver.v1)
        v2_norm = self.receiver.v2 / np.linalg.norm(self.receiver.v2)
        v3_norm = np.cross(v2_norm, v1_norm)
        R = np.array([v1_norm, v2_norm, v3_norm]).T
        
        # Transform points to local coordinates
        local_points = np.array([np.dot(R.T, pt - receiver_center) for pt in hit_points])
        
        return local_points[:, 0], local_points[:, 1]
    
    def _project_to_cylindrical_receiver(self, hit_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project points onto cylindrical receiver"""
        # Translate points relative to cylinder center
        translated_pts = hit_points - self.receiver.center
        
        # Calculate cylinder axis
        cylinder_axis = np.cross(self.receiver.base_z, self.receiver.base_x)
        
        # Project points onto cylinder surface
        projection = translated_pts - np.outer(np.dot(translated_pts, cylinder_axis), cylinder_axis)
        
        # Calculate cylindrical coordinates
        theta = np.arctan2(translated_pts[:, 1], translated_pts[:, 0])
        z = np.dot(translated_pts, cylinder_axis)
        
        # Filter out points on caps
        cap_tolerance = 1e-6
        is_on_cap = np.logical_or(
            np.abs(z - np.min(z)) < cap_tolerance,
            np.abs(z - np.max(z)) < cap_tolerance
        )
        valid_points = ~is_on_cap
        
        # Return valid points
        return theta[valid_points], z[valid_points]
    
    def calculate_flux_map(self, hit_points: np.ndarray) -> np.ndarray:
        """
        Calculate flux map from hit points
        
        Args:
            hit_points: Array of hit points in global coordinates
            
        Returns:
            2D array containing flux values
        """
        # Project points to receiver surface
        x_local, y_local = self.project_to_receiver(hit_points)
        
        # Calculate bin indices
        if self.receiver.type == "FLAT":
            dim_x = np.linalg.norm(self.receiver.v1)
            dim_y = np.linalg.norm(self.receiver.v2)
            bins_x = np.floor((x_local + dim_x/2) * self.nx / dim_x).astype(int)
            bins_y = np.floor((y_local + dim_y/2) * self.ny / dim_y).astype(int)
        else:  # CYLINDRICAL
            bins_x = np.floor(x_local * self.nx / (2 * np.pi)).astype(int)
            bins_y = np.floor((y_local + self.receiver.height/2) * self.ny / self.receiver.height).astype(int)
        
        # Ensure indices are within bounds
        bins_x = np.clip(bins_x, 0, self.nx-1)
        bins_y = np.clip(bins_y, 0, self.ny-1)
        
        # Calculate power per ray
        power_per_ray = self.sun_params.dni * self.sun_params.sun_shape_area / self.sun_params.num_rays
        
        # Calculate bin area
        if self.receiver.type == "FLAT":
            bin_area = (dim_x / self.nx) * (dim_y / self.ny)
        else:
            bin_area = (2 * np.pi * self.receiver.radius / self.nx) * (self.receiver.height / self.ny)
        
        # Calculate power per bin
        power_per_bin = power_per_ray / bin_area
        
        # Create flux map
        flux_map = np.zeros((self.nx, self.ny))
        for i in range(len(bins_x)):
            flux_map[bins_x[i], bins_y[i]] += power_per_bin
        
        return flux_map
    
    def plot_flux_map(self, flux_map: np.ndarray, save_path: Optional[str] = None):
        """
        Plot flux map
        
        Args:
            flux_map: 2D array containing flux values
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate metrics
        peak_flux = np.max(flux_map)
        avg_flux = np.mean(flux_map[flux_map > 0])
        
        # Create meshgrid for plotting
        if self.receiver.type == "FLAT":
            dim_x = np.linalg.norm(self.receiver.v1)
            dim_y = np.linalg.norm(self.receiver.v2)
            x = np.linspace(-dim_x/2, dim_x/2, self.nx)
            y = np.linspace(-dim_y/2, dim_y/2, self.ny)
        else:
            x = np.linspace(0, 2*np.pi*self.receiver.radius, self.nx)
            y = np.linspace(-self.receiver.height/2, self.receiver.height/2, self.ny)
        
        X, Y = np.meshgrid(x, y)
        
        # Plot
        plt.contourf(X, Y, flux_map.T, levels=50, cmap='jet')
        plt.colorbar(label='Flux (W/m²)')
        plt.title(f'Flux Map\nPeak: {peak_flux:.0f} W/m² | Mean: {avg_flux:.0f} W/m²')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Example usage
    # Define receiver geometry
    receiver = ReceiverGeometry(
        type="CYLINDRICAL",
        radius=9.0,
        height=22.0,
        center=np.array([0.0, 0.0, 195.0]),
        base_x=np.array([1.0, 0.0, 0.0]),
        base_z=np.array([0.0, -1.0, 0.0])
    )
    
    # Define sun parameters
    sun_params = SunParameters(
        dni=1000.0,  # W/m²
        sun_shape_area=903.318 * 604.174,  # m²
        num_rays=4380238
    )
    
    # Create calculator
    calculator = FluxMapCalculator(receiver, sun_params, grid_size=(100, 100))
    
    # Read hit points
    folder_dir = "C:/Users/fang/Documents/NREL_SOLAR/optix/build/bin/Release/"
    # folder_dir = "C:/Users/fang/Documents/NREL_SOLAR/optix/ground_truth/build/bin/Release/"

    filename = folder_dir + "output_large_system_flat_heliostats_cylindrical_receiver_stinput-sun_shape_on.csv"
    hit_points = calculator.read_hit_points(filename, solver="optix")
    
    # Calculate flux map
    flux_map = calculator.calculate_flux_map(hit_points)
    
    # Plot results
    calculator.plot_flux_map(flux_map)

if __name__ == "__main__":
    main() 
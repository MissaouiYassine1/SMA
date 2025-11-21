"""
SMA Streamlit App - Optimized Version
Slime Mould Algorithm implementation + Streamlit UI
Features:
 - Vectorized, efficient SMA core with numpy
 - Optional numba acceleration
 - Multiple benchmark functions
 - Advanced optimization techniques
 - Improved Streamlit interface
 - Performance monitoring
"""

import time
import math
from typing import Callable, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass, field
import json

# Configuration
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# -----------------------------
# Optimized Benchmark Functions
# -----------------------------

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def rastrigin(x: np.ndarray) -> float:
    n = x.size
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x))

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2)

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def ackley(x: np.ndarray) -> float:
    a, b, c = 20, 0.2, 2 * np.pi
    n = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(s1/n)) - np.exp(s2/n) + a + np.e

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def griewank(x: np.ndarray) -> float:
    return 1 + np.sum(x**2)/4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def schwefel(x: np.ndarray) -> float:
    n = x.size
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

BENCHMARKS: Dict[str, Dict[str, Any]] = {
    'Sphere': {'func': sphere, 'bounds': (-5.0, 5.0), 'global_min': 0.0},
    'Rastrigin': {'func': rastrigin, 'bounds': (-5.12, 5.12), 'global_min': 0.0},
    'Rosenbrock': {'func': rosenbrock, 'bounds': (-2.048, 2.048), 'global_min': 0.0},
    'Ackley': {'func': ackley, 'bounds': (-32.768, 32.768), 'global_min': 0.0},
    'Griewank': {'func': griewank, 'bounds': (-600, 600), 'global_min': 0.0},
    'Schwefel': {'func': schwefel, 'bounds': (-500, 500), 'global_min': 0.0},
}

# -----------------------------
# Core SMA Implementation
# -----------------------------

@dataclass
class SMAConfig:
    """Configuration class for SMA parameters"""
    population_size: int = 50
    max_iter: int = 500
    minimization: bool = True
    seed: Optional[int] = None
    early_stop: int = 100
    z_param: float = 0.03
    perturbation_scale: float = 1.0

@dataclass
class SMAResult:
    """Result container for SMA optimization"""
    best_position: np.ndarray
    best_fitness: float
    history: np.ndarray
    population: np.ndarray
    fitness: np.ndarray
    iterations: int
    execution_time: float
    convergence_iter: int

class SMAOptimizer:
    """High-performance SMA implementation with multiple optimization techniques"""
    
    def __init__(self, config: SMAConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
    def _initialize_population(self, dim: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Initialize population using Latin Hypercube Sampling for better coverage"""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            
        # Latin Hypercube Sampling for better initial coverage
        samples = np.zeros((self.config.population_size, dim))
        for i in range(dim):
            samples[:, i] = self.rng.permutation(self.config.population_size)
        samples = (samples + 0.5) / self.config.population_size
        
        return lb + (ub - lb) * samples
    
    def _evaluate_population(self, population: np.ndarray, obj_fun: Callable) -> np.ndarray:
        """Vectorized population evaluation"""
        return np.array([obj_fun(ind) for ind in population])
    
    def _compute_weights(self, fitness: np.ndarray) -> np.ndarray:
        """Compute adaptive weights based on fitness ranking"""
        sorted_idx = np.argsort(fitness)
        ranks = np.zeros_like(fitness)
        ranks[sorted_idx] = np.arange(len(fitness))
        
        # Adaptive weight calculation
        if fitness[sorted_idx[0]] == fitness[sorted_idx[-1]]:
            return np.ones_like(fitness)
        
        # Normalized weights based on rank
        weights = 1 - ranks / (len(fitness) - 1)
        return weights ** 2  # Quadratic weighting for stronger selection pressure
    
    def optimize(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray) -> SMAResult:
        """Main optimization loop"""
        start_time = time.time()
        
        # Initialize
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        population = self._initialize_population(dim, lb, ub)
        fitness = self._evaluate_population(population, obj_fun)
        
        if not self.config.minimization:
            fitness = -fitness
            
        # Track best solution
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        history = []
        stagnation_count = 0
        convergence_iter = 0
        
        # Main optimization loop
        for iteration in range(self.config.max_iter):
            # Sort population by fitness
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Update best solution
            current_best_fitness = fitness[0]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_position = population[0].copy()
                stagnation_count = 0
                convergence_iter = iteration
            else:
                stagnation_count += 1
                
            history.append(best_fitness if self.config.minimization else -best_fitness)
            
            # Early stopping
            if stagnation_count >= self.config.early_stop:
                break
                
            # Adaptive parameter z
            progress = 1.0 - (iteration / self.config.max_iter)
            z = self.config.z_param * progress
            
            # Compute weights
            weights = self._compute_weights(fitness)
            
            # Generate new positions
            new_population = self._update_positions(
                population, best_position, weights, z, lb, ub
            )
            
            # Evaluate new population
            new_fitness = self._evaluate_population(new_population, obj_fun)
            if not self.config.minimization:
                new_fitness = -new_fitness
                
            # Greedy selection
            improve_mask = new_fitness < fitness
            population[improve_mask] = new_population[improve_mask]
            fitness[improve_mask] = new_fitness[improve_mask]
            
        execution_time = time.time() - start_time
        
        return SMAResult(
            best_position=best_position,
            best_fitness=best_fitness if self.config.minimization else -best_fitness,
            history=np.array(history),
            population=population,
            fitness=fitness,
            iterations=iteration + 1,
            execution_time=execution_time,
            convergence_iter=convergence_iter
        )
    
    def _update_positions(self, population: np.ndarray, best_position: np.ndarray,
                         weights: np.ndarray, z: float, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Vectorized position update with boundary handling"""
        pop_size, dim = population.shape
        
        # Generate random components
        r1 = self.rng.rand(pop_size, dim)
        r2 = self.rng.rand(pop_size, dim)
        
        # Select random partners
        partner_indices = self.rng.randint(0, pop_size, size=pop_size)
        partners = population[partner_indices]
        
        # SMA position update
        move_toward_best = weights[:, None] * (best_position - population) * r1
        move_toward_partner = (partners - population) * r2
        
        new_positions = population + z * move_toward_best + (1 - z) * move_toward_partner
        
        # Boundary handling with random reinitialization
        out_of_bounds = (new_positions < lb) | (new_positions > ub)
        if np.any(out_of_bounds):
            for i in range(pop_size):
                for d in range(dim):
                    if new_positions[i, d] < lb[d] or new_positions[i, d] > ub[d]:
                        new_positions[i, d] = lb[d] + (ub[d] - lb[d]) * self.rng.rand()
        
        return new_positions

# -----------------------------
# Streamlit UI
# -----------------------------

def setup_sidebar():
    """Configure sidebar parameters"""
    st.sidebar.title("SMA Parameters")
    
    with st.sidebar.form("optimization_params"):
        # Problem configuration
        func_name = st.selectbox("Benchmark Function", list(BENCHMARKS.keys()))
        dim = st.number_input("Dimension", min_value=1, max_value=100, value=10, step=1)
        
        # Algorithm parameters
        pop_size = st.number_input("Population Size", min_value=10, max_value=500, value=50, step=10)
        max_iter = st.number_input("Max Iterations", min_value=50, max_value=5000, value=500, step=50)
        minimization = st.selectbox("Problem Type", ["Minimize", "Maximize"]) == "Minimize"
        
        # Advanced parameters
        st.subheader("Advanced Parameters")
        early_stop = st.number_input("Early Stopping", min_value=1, max_value=500, value=100, step=10)
        z_param = st.slider("Z Parameter", 0.001, 0.1, 0.03, 0.001)
        seed = st.number_input("Random Seed (0 for random)", value=42, min_value=0)
        
        # Visualization options
        st.subheader("Visualization")
        show_convergence = st.checkbox("Show Convergence Plot", value=True)
        show_population = st.checkbox("Show Population", value=True)
        show_statistics = st.checkbox("Show Statistics", value=True)
        
        submitted = st.form_submit_button("Run Optimization")
    
    return {
        'func_name': func_name,
        'dim': dim,
        'pop_size': pop_size,
        'max_iter': max_iter,
        'minimization': minimization,
        'early_stop': early_stop,
        'z_param': z_param,
        'seed': seed if seed != 0 else None,
        'show_convergence': show_convergence,
        'show_population': show_population,
        'show_statistics': show_statistics,
        'submitted': submitted
    }

def display_results(result: SMAResult, params: dict):
    """Display optimization results"""
    st.header("Optimization Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Fitness", f"{result.best_fitness:.6e}")
    with col2:
        st.metric("Iterations", result.iterations)
    with col3:
        st.metric("Execution Time", f"{result.execution_time:.2f}s")
    with col4:
        st.metric("Convergence Iter", result.convergence_iter)
    
    # Best solution
    st.subheader("Best Solution")
    st.write(f"First 10 dimensions: {result.best_position[:10]}")
    
    # Convergence plot
    if params['show_convergence']:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(result.history, linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Fitness (log scale)")
        ax.set_title("Convergence History")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Population visualization (for 2D problems)
    if params['dim'] == 2 and params['show_population']:
        st.subheader("Population Distribution")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        pop = result.population
        ax.scatter(pop[:, 0], pop[:, 1], c=result.fitness, 
                  cmap='viridis', s=50, alpha=0.6, label='Population')
        ax.scatter(result.best_position[0], result.best_position[1], 
                  color='red', s=200, marker='*', label='Best Solution')
        
        bounds = BENCHMARKS[params['func_name']]['bounds']
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[0], bounds[1])
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title("Final Population Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Statistics
    if params['show_statistics']:
        st.subheader("Optimization Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Fitness Statistics:")
            fitness_data = result.fitness if params['minimization'] else -result.fitness
            st.write(f"Min: {np.min(fitness_data):.6e}")
            st.write(f"Max: {np.max(fitness_data):.6e}")
            st.write(f"Mean: {np.mean(fitness_data):.6e}")
            st.write(f"Std: {np.std(fitness_data):.6e}")
        
        with col2:
            st.write("Convergence Analysis:")
            st.write(f"Final Improvement: {result.history[0] - result.history[-1]:.6e}")
            st.write(f"Stagnation Iterations: {result.iterations - result.convergence_iter}")

def export_results(result: SMAResult, params: dict):
    """Export results to files"""
    st.sidebar.header("Export Results")
    
    if st.sidebar.button("Save Results to JSON"):
        export_data = {
            'best_fitness': float(result.best_fitness),
            'best_position': result.best_position.tolist(),
            'iterations': result.iterations,
            'execution_time': result.execution_time,
            'parameters': params,
            'convergence_history': result.history.tolist()
        }
        
        timestamp = int(time.time())
        filename = f"sma_results_{params['func_name']}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        st.sidebar.success(f"Results saved to {filename}")
    
    if st.sidebar.button("Save Convergence Data to CSV"):
        df = pd.DataFrame({
            'iteration': range(len(result.history)),
            'best_fitness': result.history
        })
        
        timestamp = int(time.time())
        filename = f"sma_convergence_{params['func_name']}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        st.sidebar.success(f"Convergence data saved to {filename}")

def main():
    """Main application"""
    st.set_page_config(
        page_title="Advanced SMA Optimizer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Advanced Slime Mould Algorithm Optimizer")
    st.markdown("""
    High-performance implementation of the Slime Mould Algorithm for global optimization.
    Features vectorized operations, adaptive parameters, and comprehensive analysis tools.
    """)
    
    # Setup sidebar and get parameters
    params = setup_sidebar()
    
    if params['submitted']:
        # Get benchmark function and bounds
        benchmark = BENCHMARKS[params['func_name']]
        obj_fun = benchmark['func']
        bounds = benchmark['bounds']
        
        # Create configuration
        config = SMAConfig(
            population_size=params['pop_size'],
            max_iter=params['max_iter'],
            minimization=params['minimization'],
            seed=params['seed'],
            early_stop=params['early_stop'],
            z_param=params['z_param']
        )
        
        # Initialize optimizer
        optimizer = SMAOptimizer(config)
        
        # Run optimization
        with st.spinner("Running optimization..."):
            lb = np.full(params['dim'], bounds[0])
            ub = np.full(params['dim'], bounds[1])
            
            result = optimizer.optimize(obj_fun, params['dim'], lb, ub)
        
        # Display results
        display_results(result, params)
        
        # Export functionality
        export_results(result, params)
        
        # Performance comparison
        st.sidebar.header("Performance Info")
        st.sidebar.write(f"Function: {params['func_name']}")
        st.sidebar.write(f"Dimension: {params['dim']}D")
        st.sidebar.write(f"Global Minimum: {benchmark['global_min']}")
        st.sidebar.write(f"Error: {abs(result.best_fitness - benchmark['global_min']):.2e}")

if __name__ == "__main__":
    main()
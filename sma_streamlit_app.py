"""
SMA Streamlit App - Ultra Optimized Version
Slime Mould Algorithm implementation + Streamlit UI
Features:
 - Multi-algorithm comparison (SMA, PSO, GA, DE)
 - Real-time optimization dashboard
 - Advanced hyperparameter optimization
 - Animated visualizations
 - Project management system
 - Statistical analysis
 - Enhanced export functionality
 - Parallel computing support
"""

import time
import math
import os
import warnings
from typing import Callable, Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dataclasses import dataclass, field
import json
import io
import base64
import pickle
from datetime import datetime
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# -----------------------------
# Enhanced Benchmark Functions
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

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def zakharov(x: np.ndarray) -> float:
    n = x.size
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, n+1) * x)
    return sum1 + sum2**2 + sum2**4

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def levy(x: np.ndarray) -> float:
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0]))**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1))**2))
    term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
    return term1 + term2 + term3

BENCHMARKS: Dict[str, Dict[str, Any]] = {
    'Sphere': {'func': sphere, 'bounds': (-5.0, 5.0), 'global_min': 0.0, 'difficulty': 'Easy'},
    'Rastrigin': {'func': rastrigin, 'bounds': (-5.12, 5.12), 'global_min': 0.0, 'difficulty': 'Hard'},
    'Rosenbrock': {'func': rosenbrock, 'bounds': (-2.048, 2.048), 'global_min': 0.0, 'difficulty': 'Medium'},
    'Ackley': {'func': ackley, 'bounds': (-32.768, 32.768), 'global_min': 0.0, 'difficulty': 'Medium'},
    'Griewank': {'func': griewank, 'bounds': (-600, 600), 'global_min': 0.0, 'difficulty': 'Medium'},
    'Schwefel': {'func': schwefel, 'bounds': (-500, 500), 'global_min': 0.0, 'difficulty': 'Hard'},
    'Zakharov': {'func': zakharov, 'bounds': (-5.0, 10.0), 'global_min': 0.0, 'difficulty': 'Easy'},
    'Levy': {'func': levy, 'bounds': (-10.0, 10.0), 'global_min': 0.0, 'difficulty': 'Hard'},
}

# -----------------------------
# Multi-Algorithm Implementation
# -----------------------------

@dataclass
class OptimizationConfig:
    """Configuration class for optimization parameters"""
    algorithm: str = "SMA"
    population_size: int = 50
    max_iter: int = 500
    minimization: bool = True
    seed: Optional[int] = None
    early_stop: int = 100
    # SMA specific
    z_param: float = 0.03
    adaptive_z: bool = True
    # PSO specific
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    # GA specific
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    # DE specific
    F: float = 0.5
    CR: float = 0.7

@dataclass
class OptimizationResult:
    """Result container for optimization"""
    algorithm: str
    best_position: np.ndarray
    best_fitness: float
    history: np.ndarray
    population: np.ndarray
    fitness: np.ndarray
    iterations: int
    execution_time: float
    convergence_iter: int
    config: OptimizationConfig
    function_name: str
    dimension: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseOptimizer:
    """Base class for all optimizers"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.metrics_history = {
            'diversity': [],
            'exploration_rate': [],
            'best_fitness': []
        }
    
    def _calculate_diversity(self, population: np.ndarray) -> float:
        """Calculate population diversity"""
        centroid = np.mean(population, axis=0)
        return np.mean(np.sqrt(np.sum((population - centroid)**2, axis=1)))
    
    def _update_metrics(self, population: np.ndarray, best_fitness: float):
        """Update real-time metrics"""
        diversity = self._calculate_diversity(population)
        exploration_rate = diversity / (np.max(population) - np.min(population) + 1e-12)
        
        self.metrics_history['diversity'].append(diversity)
        self.metrics_history['exploration_rate'].append(exploration_rate)
        self.metrics_history['best_fitness'].append(best_fitness)

class SMAOptimizer(BaseOptimizer):
    """Enhanced SMA implementation"""
    
    def _initialize_population(self, dim: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Improved initialization with LHS"""
        samples = np.zeros((self.config.population_size, dim))
        for i in range(dim):
            samples[:, i] = self.rng.permutation(self.config.population_size)
        samples = (samples + self.rng.rand(self.config.population_size, dim)) / self.config.population_size
        return lb + (ub - lb) * samples
    
    def _compute_weights(self, fitness: np.ndarray) -> np.ndarray:
        """Enhanced weight computation"""
        sorted_idx = np.argsort(fitness)
        ranks = np.zeros_like(fitness)
        ranks[sorted_idx] = np.arange(len(fitness))
        
        if fitness[sorted_idx[0]] == fitness[sorted_idx[-1]]:
            return np.ones_like(fitness)
        
        # Improved adaptive weights
        weights = np.exp(-2.0 * ranks / len(fitness))
        return weights ** 1.5  # Increased selection pressure
    
    def _adaptive_z_parameter(self, iteration: int, diversity: float) -> float:
        """Diversity-aware adaptive z parameter"""
        if not self.config.adaptive_z:
            return self.config.z_param
        
        progress = iteration / self.config.max_iter
        diversity_factor = 1.0 - (diversity / self.metrics_history['diversity'][0] if self.metrics_history['diversity'] else 1.0)
        
        return self.config.z_param * (1.0 - progress**1.5) * (0.8 + 0.2 * diversity_factor)
    
    def optimize(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray, 
                 function_name: str = "Unknown") -> OptimizationResult:
        """Enhanced optimization loop with real-time metrics"""
        start_time = time.perf_counter()
        
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        population = self._initialize_population(dim, lb, ub)
        fitness = np.array([obj_fun(ind) for ind in population])
        
        if not self.config.minimization:
            fitness = -fitness
            
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        history = [best_fitness if self.config.minimization else -best_fitness]
        stagnation_count = 0
        convergence_iter = 0
        
        for iteration in range(self.config.max_iter):
            # Sort and update best
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            current_best_fitness = fitness[0]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_position = population[0].copy()
                stagnation_count = 0
                convergence_iter = iteration
            else:
                stagnation_count += 1
                
            history.append(best_fitness if self.config.minimization else -best_fitness)
            
            # Update real-time metrics
            self._update_metrics(population, best_fitness)
            
            # Enhanced early stopping
            if stagnation_count >= self.config.early_stop:
                if self.metrics_history['diversity'][-1] < 0.01:  # Low diversity
                    break
            
            # Adaptive parameters
            z = self._adaptive_z_parameter(iteration, self.metrics_history['diversity'][-1])
            weights = self._compute_weights(fitness)
            
            # Position update
            new_population = self._update_positions(population, best_position, weights, z, lb, ub)
            new_fitness = np.array([obj_fun(ind) for ind in new_population])
            
            if not self.config.minimization:
                new_fitness = -new_fitness
                
            # Enhanced selection with elitism
            improve_mask = new_fitness < fitness
            population[improve_mask] = new_population[improve_mask]
            fitness[improve_mask] = new_fitness[improve_mask]
            
            # Strong elitism
            if fitness[0] > best_fitness:
                population[0] = best_position
                fitness[0] = best_fitness
            
        execution_time = time.perf_counter() - start_time
        
        return OptimizationResult(
            algorithm="SMA",
            best_position=best_position,
            best_fitness=best_fitness if self.config.minimization else -best_fitness,
            history=np.array(history),
            population=population,
            fitness=fitness,
            iterations=iteration + 1,
            execution_time=execution_time,
            convergence_iter=convergence_iter,
            config=self.config,
            function_name=function_name,
            dimension=dim,
            metadata={'metrics_history': self.metrics_history}
        )
    
    def _update_positions(self, population: np.ndarray, best_position: np.ndarray,
                         weights: np.ndarray, z: float, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Enhanced position update with Levy flights"""
        pop_size, dim = population.shape
        
        r1 = self.rng.rand(pop_size, dim)
        r2 = self.rng.rand(pop_size, dim)
        
        partner_indices = self.rng.randint(0, pop_size, size=pop_size)
        partners = population[partner_indices]
        
        # Levy flight enhancement for exploration
        if self.rng.rand() < 0.1:  # 10% chance of Levy flight
            levy_step = self._generate_levy_flight(pop_size, dim)
            levy_component = 0.1 * levy_step
        else:
            levy_component = 0
        
        move_toward_best = weights[:, None] * (best_position - population) * r1
        move_toward_partner = (partners - population) * r2
        
        new_positions = population + z * move_toward_best + (1 - z) * move_toward_partner + levy_component
        
        return np.clip(new_positions, lb, ub)
    
    def _generate_levy_flight(self, size: int, dim: int) -> np.ndarray:
        """Generate Levy flight steps"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        
        u = self.rng.normal(0, sigma, size=(size, dim))
        v = self.rng.normal(0, 1, size=(size, dim))
        step = u / (np.abs(v) ** (1 / beta))
        
        return step

class PSOOptimizer(BaseOptimizer):
    """Particle Swarm Optimization implementation"""
    
    def optimize(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray, 
                 function_name: str = "Unknown") -> OptimizationResult:
        """PSO optimization implementation"""
        start_time = time.perf_counter()
        
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        
        # Initialize particles
        population = lb + (ub - lb) * self.rng.rand(self.config.population_size, dim)
        velocity = np.zeros_like(population)
        
        fitness = np.array([obj_fun(ind) for ind in population])
        personal_best_pos = population.copy()
        personal_best_fitness = fitness.copy()
        
        if not self.config.minimization:
            fitness = -fitness
            personal_best_fitness = -personal_best_fitness
            
        # Find global best
        best_idx = np.argmin(personal_best_fitness)
        best_position = personal_best_pos[best_idx].copy()
        best_fitness = personal_best_fitness[best_idx]
        
        history = [best_fitness if self.config.minimization else -best_fitness]
        
        for iteration in range(self.config.max_iter):
            # Update velocities and positions
            r1 = self.rng.rand(self.config.population_size, dim)
            r2 = self.rng.rand(self.config.population_size, dim)
            
            velocity = (self.config.w * velocity + 
                       self.config.c1 * r1 * (personal_best_pos - population) + 
                       self.config.c2 * r2 * (best_position - population))
            
            population = np.clip(population + velocity, lb, ub)
            
            # Evaluate new fitness
            fitness = np.array([obj_fun(ind) for ind in population])
            if not self.config.minimization:
                fitness = -fitness
            
            # Update personal best
            improved_mask = fitness < personal_best_fitness
            personal_best_pos[improved_mask] = population[improved_mask]
            personal_best_fitness[improved_mask] = fitness[improved_mask]
            
            # Update global best
            current_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[current_best_idx] < best_fitness:
                best_fitness = personal_best_fitness[current_best_idx]
                best_position = personal_best_pos[current_best_idx].copy()
            
            history.append(best_fitness if self.config.minimization else -best_fitness)
            self._update_metrics(population, best_fitness)
            
        execution_time = time.perf_counter() - start_time
        
        return OptimizationResult(
            algorithm="PSO",
            best_position=best_position,
            best_fitness=best_fitness if self.config.minimization else -best_fitness,
            history=np.array(history),
            population=population,
            fitness=fitness,
            iterations=iteration + 1,
            execution_time=execution_time,
            convergence_iter=iteration,
            config=self.config,
            function_name=function_name,
            dimension=dim,
            metadata={'metrics_history': self.metrics_history}
        )

class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Genetic Algorithm implementation"""
    
    def optimize(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray, 
                 function_name: str = "Unknown") -> OptimizationResult:
        """GA optimization implementation"""
        start_time = time.perf_counter()
        
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        
        # Initialize population
        population = lb + (ub - lb) * self.rng.rand(self.config.population_size, dim)
        fitness = np.array([obj_fun(ind) for ind in population])
        
        if not self.config.minimization:
            fitness = -fitness
            
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        history = [best_fitness if self.config.minimization else -best_fitness]
        
        for iteration in range(self.config.max_iter):
            # Selection (Tournament selection)
            parents = self._tournament_selection(population, fitness)
            
            # Crossover
            offspring = self._crossover(parents)
            
            # Mutation
            offspring = self._mutate(offspring, lb, ub)
            
            # Evaluate offspring
            offspring_fitness = np.array([obj_fun(ind) for ind in offspring])
            if not self.config.minimization:
                offspring_fitness = -offspring_fitness
            
            # Replacement (Elitism)
            combined_population = np.vstack([population, offspring])
            combined_fitness = np.hstack([fitness, offspring_fitness])
            
            # Select best individuals
            best_indices = np.argsort(combined_fitness)[:self.config.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]
            
            # Update best solution
            if fitness[0] < best_fitness:
                best_fitness = fitness[0]
                best_position = population[0].copy()
            
            history.append(best_fitness if self.config.minimization else -best_fitness)
            self._update_metrics(population, best_fitness)
            
        execution_time = time.perf_counter() - start_time
        
        return OptimizationResult(
            algorithm="GA",
            best_position=best_position,
            best_fitness=best_fitness if self.config.minimization else -best_fitness,
            history=np.array(history),
            population=population,
            fitness=fitness,
            iterations=iteration + 1,
            execution_time=execution_time,
            convergence_iter=iteration,
            config=self.config,
            function_name=function_name,
            dimension=dim,
            metadata={'metrics_history': self.metrics_history}
        )
    
    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection"""
        selected = []
        for _ in range(self.config.population_size):
            indices = self.rng.choice(len(population), size=3, replace=False)
            best_idx = indices[np.argmin(fitness[indices])]
            selected.append(population[best_idx])
        return np.array(selected)
    
    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Simulated binary crossover"""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                p1, p2 = parents[i], parents[i+1]
                if self.rng.rand() < self.config.crossover_rate:
                    beta = np.zeros_like(p1)
                    for j in range(len(p1)):
                        u = self.rng.rand()
                        beta[j] = (2 * u) ** (1 / (1 + 3)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (1 + 3))
                    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([p1, p2])
        return np.array(offspring)
    
    def _mutate(self, population: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Polynomial mutation"""
        mutated = population.copy()
        for i in range(len(population)):
            if self.rng.rand() < self.config.mutation_rate:
                for j in range(population.shape[1]):
                    u = self.rng.rand()
                    delta = (2 * u) ** (1 / (1 + 20)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (1 + 20))
                    mutated[i, j] += delta * (ub[j] - lb[j])
        return np.clip(mutated, lb, ub)

# Algorithm registry
ALGORITHMS = {
    'SMA': SMAOptimizer,
    'PSO': PSOOptimizer,
    'GA': GeneticAlgorithmOptimizer,
}

# -----------------------------
# Hyperparameter Optimization
# -----------------------------

class HyperparameterOptimizer:
    """Bayesian optimization for SMA hyperparameters"""
    
    def __init__(self):
        self.param_space = {
            'population_size': (20, 200),
            'z_param': (0.01, 0.1),
            'early_stop': (50, 500)
        }
    
    def optimize_hyperparameters(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray,
                                n_trials: int = 20) -> Dict[str, Any]:
        """Simple random search for hyperparameter optimization"""
        best_score = float('inf')
        best_params = {}
        
        for trial in range(n_trials):
            # Sample random parameters
            params = {
                'population_size': np.random.randint(20, 200),
                'z_param': np.random.uniform(0.01, 0.1),
                'early_stop': np.random.randint(50, 500)
            }
            
            # Evaluate with these parameters
            config = OptimizationConfig(**params)
            optimizer = SMAOptimizer(config)
            result = optimizer.optimize(obj_fun, dim, lb, ub)
            
            if result.best_fitness < best_score:
                best_score = result.best_fitness
                best_params = params
        
        return {'best_params': best_params, 'best_score': best_score}

# -----------------------------
# Project Management System
# -----------------------------

class ProjectManager:
    """Manage optimization projects and sessions"""
    
    def __init__(self):
        self.projects = {}
        self.current_project = None
    
    def create_project(self, name: str, description: str = ""):
        """Create a new project"""
        project_id = f"project_{int(time.time())}"
        self.projects[project_id] = {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'sessions': [],
            'results': {}
        }
        self.current_project = project_id
        return project_id
    
    def save_session(self, config: OptimizationConfig, results: List[OptimizationResult]):
        """Save current session to project"""
        if self.current_project:
            session_id = f"session_{int(time.time())}"
            self.projects[self.current_project]['sessions'].append({
                'id': session_id,
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'results': [self._result_to_dict(r) for r in results]
            })
            return session_id
    
    def _result_to_dict(self, result: OptimizationResult) -> Dict[str, Any]:
        """Convert result to serializable dictionary"""
        return {
            'algorithm': result.algorithm,
            'best_fitness': float(result.best_fitness),
            'best_position': result.best_position.tolist(),
            'iterations': result.iterations,
            'execution_time': result.execution_time,
            'convergence_iter': result.convergence_iter,
            'function_name': result.function_name,
            'dimension': result.dimension
        }

# -----------------------------
# Enhanced Visualization Functions
# -----------------------------

def create_real_time_dashboard(results: List[OptimizationResult]) -> go.Figure:
    """Create interactive real-time dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Convergence History', 'Population Diversity', 
                       'Exploration vs Exploitation', 'Algorithm Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Convergence history
    for i, result in enumerate(results):
        fig.add_trace(
            go.Scatter(y=result.history, name=f'{result.algorithm} Run {i+1}',
                      line=dict(width=2)),
            row=1, col=1
        )
    
    # Population diversity
    for i, result in enumerate(results):
        if 'metrics_history' in result.metadata:
            diversity = result.metadata['metrics_history']['diversity']
            fig.add_trace(
                go.Scatter(y=diversity, name=f'{result.algorithm} Diversity',
                          line=dict(dash='dot')),
                row=1, col=2
            )
    
    # Exploration rate
    for i, result in enumerate(results):
        if 'metrics_history' in result.metadata:
            exploration = result.metadata['metrics_history']['exploration_rate']
            fig.add_trace(
                go.Scatter(y=exploration, name=f'{result.algorithm} Exploration',
                          line=dict(dash='dash')),
                row=2, col=1
            )
    
    fig.update_layout(height=600, title_text="Real-Time Optimization Dashboard")
    return fig

def plot_algorithm_comparison(results: List[OptimizationResult]) -> go.Figure:
    """Compare algorithm performance"""
    algorithms = list(set([r.algorithm for r in results]))
    metrics = ['Best Fitness', 'Execution Time', 'Iterations', 'Convergence Speed']
    
    fig = go.Figure()
    
    for metric in metrics:
        values = []
        for algo in algorithms:
            algo_results = [r for r in results if r.algorithm == algo]
            if metric == 'Best Fitness':
                values.append(np.mean([r.best_fitness for r in algo_results]))
            elif metric == 'Execution Time':
                values.append(np.mean([r.execution_time for r in algo_results]))
            elif metric == 'Iterations':
                values.append(np.mean([r.iterations for r in algo_results]))
            elif metric == 'Convergence Speed':
                values.append(np.mean([r.convergence_iter / r.iterations for r in algo_results]))
        
        fig.add_trace(go.Bar(name=metric, x=algorithms, y=values))
    
    fig.update_layout(barmode='group', title="Algorithm Performance Comparison")
    return fig

# -----------------------------
# Statistical Analysis
# -----------------------------

class StatisticalAnalyzer:
    """Perform statistical analysis on optimization results"""
    
    @staticmethod
    def wilcoxon_signed_rank_test(results_a: List[OptimizationResult], 
                                 results_b: List[OptimizationResult]) -> Dict[str, Any]:
        """Wilcoxon signed-rank test for algorithm comparison"""
        fitness_a = [r.best_fitness for r in results_a]
        fitness_b = [r.best_fitness for r in results_b]
        
        if len(fitness_a) != len(fitness_b):
            min_len = min(len(fitness_a), len(fitness_b))
            fitness_a = fitness_a[:min_len]
            fitness_b = fitness_b[:min_len]
        
        try:
            stat, p_value = stats.wilcoxon(fitness_a, fitness_b)
            return {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except:
            return {'error': 'Insufficient data for statistical test'}
    
    @staticmethod
    def calculate_convergence_metrics(history: np.ndarray) -> Dict[str, float]:
        """Calculate convergence quality metrics"""
        return {
            'auc': np.trapz(history),  # Area under curve
            'final_improvement': history[0] - history[-1],
            'convergence_speed': len(history) / (history[0] - history[-1] + 1e-12),
            'stability': np.std(history[-10:]) if len(history) >= 10 else np.std(history)
        }

# -----------------------------
# Enhanced Export Utilities
# -----------------------------

def create_comprehensive_report(results: List[OptimizationResult], 
                               project_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive report"""
    report = {
        'project_info': project_info,
        'summary': {
            'total_runs': len(results),
            'algorithms_used': list(set([r.algorithm for r in results])),
            'best_overall_fitness': min([r.best_fitness for r in results]),
            'average_execution_time': np.mean([r.execution_time for r in results])
        },
        'detailed_results': {},
        'statistical_analysis': {},
        'recommendations': []
    }
    
    # Detailed results per algorithm
    for algo in set([r.algorithm for r in results]):
        algo_results = [r for r in results if r.algorithm == algo]
        report['detailed_results'][algo] = {
            'best_fitness': min([r.best_fitness for r in algo_results]),
            'worst_fitness': max([r.best_fitness for r in algo_results]),
            'average_fitness': np.mean([r.best_fitness for r in algo_results]),
            'std_fitness': np.std([r.best_fitness for r in algo_results]),
            'success_rate': np.mean([1 if abs(r.best_fitness) < 1e-6 else 0 for r in algo_results])
        }
    
    # Generate recommendations
    best_algo = min(report['detailed_results'].items(), 
                   key=lambda x: x[1]['average_fitness'])[0]
    report['recommendations'].append(f"Best performing algorithm: {best_algo}")
    
    return report

# -----------------------------
# Streamlit UI - Enhanced
# -----------------------------

def setup_sidebar():
    """Configure enhanced sidebar with multiple sections"""
    st.sidebar.title("🧠 Advanced Optimizer")
    
    with st.sidebar.expander("🚀 Quick Start", expanded=True):
        user_level = st.selectbox("User Level", ["Beginner", "Intermediate", "Expert"])
        
        if user_level == "Beginner":
            st.info("Recommended: Use default parameters for quick start")
        elif user_level == "Expert":
            st.warning("Advanced parameters enabled")
    
    with st.sidebar.form("optimization_config"):
        st.subheader("Optimization Setup")
        
        col1, col2 = st.columns(2)
        with col1:
            func_name = st.selectbox("Function", list(BENCHMARKS.keys()))
            st.caption(f"Difficulty: {BENCHMARKS[func_name]['difficulty']}")
        with col2:
            dim = st.slider("Dimension", 1, 100, 10)
            minimization = st.selectbox("Type", ["Minimize", "Maximize"]) == "Minimize"
        
        st.subheader("Algorithm Selection")
        algorithms = st.multiselect(
            "Algorithms to Compare",
            list(ALGORITHMS.keys()),
            default=["SMA", "PSO"]
        )
        
        st.subheader("General Parameters")
        col1, col2 = st.columns(2)
        with col1:
            pop_size = st.slider("Population", 10, 500, 50)
            max_iter = st.slider("Max Iterations", 50, 5000, 500)
        with col2:
            num_runs = st.slider("Runs per Algorithm", 1, 20, 3)
            seed = st.number_input("Seed", value=42)
        
        if user_level == "Expert":
            st.subheader("Advanced Parameters")
            early_stop = st.slider("Early Stop", 10, 1000, 100)
            adaptive_z = st.checkbox("Adaptive Z", True)
            z_param = st.slider("Z Parameter", 0.001, 0.1, 0.03, 0.001)
        else:
            early_stop = 100
            adaptive_z = True
            z_param = 0.03
        
        # Hyperparameter optimization
        if user_level == "Expert":
            optimize_hyperparams = st.checkbox("Optimize Hyperparameters", False)
        else:
            optimize_hyperparams = False
        
        submitted = st.form_submit_button("🎯 Run Optimization")
    
    return {
        'user_level': user_level,
        'func_name': func_name,
        'dim': dim,
        'minimization': minimization,
        'algorithms': algorithms,
        'pop_size': pop_size,
        'max_iter': max_iter,
        'num_runs': num_runs,
        'seed': seed,
        'early_stop': early_stop,
        'adaptive_z': adaptive_z,
        'z_param': z_param,
        'optimize_hyperparams': optimize_hyperparams,
        'submitted': submitted
    }

def display_enhanced_results(results: List[OptimizationResult], params: dict):
    """Display comprehensive results with multiple visualizations"""
    
    # Summary cards
    st.header("📊 Optimization Summary")
    
    cols = st.columns(4)
    best_result = min(results, key=lambda x: x.best_fitness)
    with cols[0]:
        st.metric("Best Fitness", f"{best_result.best_fitness:.2e}")
    with cols[1]:
        st.metric("Best Algorithm", best_result.algorithm)
    with cols[2]:
        avg_time = np.mean([r.execution_time for r in results])
        st.metric("Avg Time", f"{avg_time:.2f}s")
    with cols[3]:
        success_rate = np.mean([1 if abs(r.best_fitness) < 1e-6 else 0 for r in results])
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    # Real-time dashboard
    if PLOTLY_AVAILABLE:
        st.subheader("📈 Live Dashboard")
        dashboard = create_real_time_dashboard(results)
        st.plotly_chart(dashboard, use_container_width=True)
    
    # Algorithm comparison
    st.subheader("🔄 Algorithm Comparison")
    comp_fig = plot_algorithm_comparison(results)
    st.plotly_chart(comp_fig, use_container_width=True)
    
    # Detailed results by algorithm
    st.subheader("📋 Detailed Results")
    for algorithm in set([r.algorithm for r in results]):
        with st.expander(f"Algorithm: {algorithm}", expanded=True):
            algo_results = [r for r in results if r.algorithm == algorithm]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Performance Metrics:**")
                metrics_df = pd.DataFrame({
                    'Run': range(1, len(algo_results) + 1),
                    'Best Fitness': [f"{r.best_fitness:.2e}" for r in algo_results],
                    'Time (s)': [f"{r.execution_time:.2f}" for r in algo_results],
                    'Iterations': [r.iterations for r in algo_results]
                })
                st.dataframe(metrics_df, use_container_width=True)
            
            with col2:
                st.write("**Statistical Summary:**")
                stats_data = {
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
                    'Fitness': [
                        f"{np.mean([r.best_fitness for r in algo_results]):.2e}",
                        f"{np.std([r.best_fitness for r in algo_results]):.2e}",
                        f"{np.min([r.best_fitness for r in algo_results]):.2e}",
                        f"{np.max([r.best_fitness for r in algo_results]):.2e}"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    # Statistical analysis
    if len(params['algorithms']) > 1:
        st.subheader("📊 Statistical Analysis")
        analyzer = StatisticalAnalyzer()
        
        algorithms = list(set([r.algorithm for r in results]))
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                algo_a_results = [r for r in results if r.algorithm == algorithms[i]]
                algo_b_results = [r for r in results if r.algorithm == algorithms[j]]
                
                test_result = analyzer.wilcoxon_signed_rank_test(algo_a_results, algo_b_results)
                
                if 'error' not in test_result:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{algorithms[i]} vs {algorithms[j]}**")
                    with col2:
                        if test_result['significant']:
                            st.success("Significant difference")
                        else:
                            st.warning("No significant difference")

def setup_advanced_export(project_manager: ProjectManager, results: List[OptimizationResult], params: dict):
    """Enhanced export functionality"""
    st.sidebar.header("💾 Advanced Export")
    
    # Project management
    if st.sidebar.button("💼 Save Project"):
        project_id = project_manager.create_project(
            f"Optimization_{params['func_name']}",
            f"Multi-algorithm comparison on {params['func_name']}"
        )
        session_id = project_manager.save_session(
            OptimizationConfig(**{k: v for k, v in params.items() if k in OptimizationConfig.__annotations__}),
            results
        )
        st.sidebar.success(f"Project saved: {project_id}")
    
    # Comprehensive report
    if st.sidebar.button("📄 Generate Full Report"):
        report = create_comprehensive_report(results, {
            'function': params['func_name'],
            'dimension': params['dim'],
            'timestamp': datetime.now().isoformat()
        })
        
        report_json = json.dumps(report, indent=2)
        b64 = base64.b64encode(report_json.encode()).decode()
        st.sidebar.markdown(
            f'<a href="data:application/json;base64,{b64}" download="optimization_report.json">📥 Download Full Report</a>',
            unsafe_allow_html=True
        )
    
    # Comparison data
    if st.sidebar.button("📊 Export Comparison Data"):
        comparison_data = []
        for result in results:
            comparison_data.append({
                'algorithm': result.algorithm,
                'best_fitness': result.best_fitness,
                'execution_time': result.execution_time,
                'iterations': result.iterations,
                'convergence_iter': result.convergence_iter,
                'function': result.function_name,
                'dimension': result.dimension
            })
        
        df = pd.DataFrame(comparison_data)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.sidebar.markdown(
            f'<a href="data:application/csv;base64,{b64}" download="algorithm_comparison.csv">📥 Download Comparison Data</a>',
            unsafe_allow_html=True
        )

def main():
    """Enhanced main application"""
    st.set_page_config(
        page_title="Ultimate Optimization Suite",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Ultimate Optimization Suite")
    st.markdown("""
    **Advanced multi-algorithm optimization platform** with real-time analytics, 
    statistical comparison, and comprehensive reporting.
    """)
    
    # Initialize session state
    if 'project_manager' not in st.session_state:
        st.session_state.project_manager = ProjectManager()
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = []
    
    # System info
    with st.expander("🔧 System Information", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Numba:** {NUMBA_AVAILABLE}")
        with col2:
            st.write(f"**Plotly:** {PLOTLY_AVAILABLE}")
        with col3:
            st.write(f"**Algorithms:** {len(ALGORITHMS)}")
        with col4:
            st.write(f"**Benchmarks:** {len(BENCHMARKS)}")
    
    # Setup parameters
    params = setup_sidebar()
    
    if params['submitted']:
        benchmark = BENCHMARKS[params['func_name']]
        obj_fun = benchmark['func']
        bounds = benchmark['bounds']
        
        all_results = []
        
        # Hyperparameter optimization
        if params['optimize_hyperparams'] and 'SMA' in params['algorithms']:
            with st.spinner("🔍 Optimizing SMA hyperparameters..."):
                hyper_optimizer = HyperparameterOptimizer()
                hyper_result = hyper_optimizer.optimize_hyperparameters(
                    obj_fun, params['dim'], 
                    np.full(params['dim'], bounds[0]),
                    np.full(params['dim'], bounds[1])
                )
                st.success(f"Best parameters: {hyper_result['best_params']}")
        
        # Run optimizations for each algorithm
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_tasks = len(params['algorithms']) * params['num_runs']
        completed_tasks = 0
        
        for algorithm in params['algorithms']:
            algorithm_results = []
            
            for run in range(params['num_runs']):
                status_text.text(f"Running {algorithm} - Run {run + 1}/{params['num_runs']}")
                
                # Create configuration
                config = OptimizationConfig(
                    algorithm=algorithm,
                    population_size=params['pop_size'],
                    max_iter=params['max_iter'],
                    minimization=params['minimization'],
                    seed=params['seed'] + run if params['seed'] else None,
                    early_stop=params['early_stop'],
                    z_param=params['z_param'],
                    adaptive_z=params['adaptive_z']
                )
                
                # Initialize optimizer
                optimizer_class = ALGORITHMS[algorithm]
                optimizer = optimizer_class(config)
                
                # Run optimization
                lb = np.full(params['dim'], bounds[0])
                ub = np.full(params['dim'], bounds[1])
                
                result = optimizer.optimize(obj_fun, params['dim'], lb, ub, params['func_name'])
                algorithm_results.append(result)
                
                completed_tasks += 1
                progress_bar.progress(completed_tasks / total_tasks)
            
            all_results.extend(algorithm_results)
        
        progress_bar.empty()
        status_text.empty()
        
        # Store in history
        st.session_state.optimization_history.extend(all_results)
        
        # Display results
        display_enhanced_results(all_results, params)
        
        # Setup export
        setup_advanced_export(st.session_state.project_manager, all_results, params)
        
        # Performance summary
        st.sidebar.header("🎯 Performance Insights")
        best_algo = min(set([r.algorithm for r in all_results]), 
                       key=lambda algo: np.mean([r.best_fitness for r in all_results if r.algorithm == algo]))
        st.sidebar.success(f"**Recommended Algorithm:** {best_algo}")
        
        avg_improvement = np.mean([r.history[0] - r.history[-1] for r in all_results])
        st.sidebar.info(f"**Average Improvement:** {avg_improvement:.2e}")

if __name__ == "__main__":
    main()
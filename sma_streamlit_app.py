"""
SMA Streamlit App - Version Ultra Optimisée et Améliorée
Slime Mould Algorithm implementation + Streamlit UI
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
import concurrent.futures
from multiprocessing import cpu_count
import hashlib
from functools import lru_cache
import logging
from logging.handlers import RotatingFileHandler

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
# Système de Cache et Mémoire
# -----------------------------

class ResultsCache:
    """Cache intelligent pour éviter les recalculs"""
    
    def __init__(self, max_size=50):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key):
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[min_key]
            del self.access_count[min_key]
        self.cache[key] = value
        self.access_count[key] = 1

function_cache = {}

# -----------------------------
# Système de Logging et Monitoring
# -----------------------------

class OptimizationLogger:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('OptimizationSuite')
        self.logger.setLevel(log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        file_handler = RotatingFileHandler(
            'optimization_suite.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_optimization_start(self, config: 'OptimizationConfig'):
        self.logger.info(f"Starting optimization with {config.algorithm}")
    
    def log_optimization_end(self, result: 'OptimizationResult'):
        self.logger.info(
            f"Optimization completed: {result.algorithm} - "
            f"Best fitness: {result.best_fitness:.6f}"
        )

class RealTimeMonitor:
    def __init__(self):
        self.metrics = {
            'function_evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'execution_times': [],
            'memory_usage': [],
            'diversity_history': []
        }
    
    def log_function_evaluation(self):
        self.metrics['function_evaluations'] += 1
    
    def log_cache_hit(self):
        self.metrics['cache_hits'] += 1
    
    def log_cache_miss(self):
        self.metrics['cache_misses'] += 1
    
    def log_execution_time(self, time_taken):
        self.metrics['execution_times'].append(time_taken)
    
    def log_diversity(self, diversity):
        self.metrics['diversity_history'].append(diversity)
    
    def get_cache_efficiency(self):
        total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        return self.metrics['cache_hits'] / total if total > 0 else 0
    
    def display_metrics(self):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Function Evaluations", self.metrics['function_evaluations'])
        with col2:
            st.metric("Cache Efficiency", f"{self.get_cache_efficiency():.1%}")
        with col3:
            avg_time = np.mean(self.metrics['execution_times']) if self.metrics['execution_times'] else 0
            st.metric("Avg Time/Run", f"{avg_time:.2f}s")
        with col4:
            st.metric("Total Runs", len(self.metrics['execution_times']))

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

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def happy_cat(x: np.ndarray) -> float:
    n = x.size
    sum_sq = np.sum(x**2)
    sum_quad = np.sum(x**4)
    return ((sum_sq - n)**2)**0.125 + (0.5 * sum_sq + sum_quad) / n + 0.5

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def alpine(x: np.ndarray) -> float:
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

@njit(fastmath=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
def michalewicz(x: np.ndarray) -> float:
    m = 10
    result = 0
    for i in range(len(x)):
        result -= np.sin(x[i]) * np.sin((i + 1) * x[i]**2 / np.pi)**(2 * m)
    return result

BENCHMARKS: Dict[str, Dict[str, Any]] = {
    'Sphere': {'func': sphere, 'bounds': (-5.0, 5.0), 'global_min': 0.0, 'difficulty': 'Easy'},
    'Rastrigin': {'func': rastrigin, 'bounds': (-5.12, 5.12), 'global_min': 0.0, 'difficulty': 'Hard'},
    'Rosenbrock': {'func': rosenbrock, 'bounds': (-2.048, 2.048), 'global_min': 0.0, 'difficulty': 'Medium'},
    'Ackley': {'func': ackley, 'bounds': (-32.768, 32.768), 'global_min': 0.0, 'difficulty': 'Medium'},
    'Griewank': {'func': griewank, 'bounds': (-600, 600), 'global_min': 0.0, 'difficulty': 'Medium'},
    'Schwefel': {'func': schwefel, 'bounds': (-500, 500), 'global_min': 0.0, 'difficulty': 'Hard'},
    'Zakharov': {'func': zakharov, 'bounds': (-5.0, 10.0), 'global_min': 0.0, 'difficulty': 'Easy'},
    'Levy': {'func': levy, 'bounds': (-10.0, 10.0), 'global_min': 0.0, 'difficulty': 'Hard'},
    'HappyCat': {'func': happy_cat, 'bounds': (-2.0, 2.0), 'global_min': 0.0, 'difficulty': 'Hard'},
    'Alpine': {'func': alpine, 'bounds': (-10.0, 10.0), 'global_min': 0.0, 'difficulty': 'Medium'},
    'Michalewicz': {'func': michalewicz, 'bounds': (0.0, np.pi), 'global_min': -9.66, 'difficulty': 'Hard'},
}

# -----------------------------
# Configuration et Résultats
# -----------------------------

@dataclass
class OptimizationConfig:
    algorithm: str = "SMA"
    population_size: int = 50
    max_iter: int = 500
    minimization: bool = True
    seed: Optional[int] = None
    early_stop: int = 100
    z_param: float = 0.03
    adaptive_z: bool = True
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    F: float = 0.5
    CR: float = 0.7

@dataclass
class OptimizationResult:
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

# -----------------------------
# Système de Parallelisation SIMPLIFIÉ
# -----------------------------

class ParallelOptimizer:
    """Exécution parallèle des algorithmes avec ThreadPoolExecutor"""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or min(cpu_count(), 4)  # Réduit pour stabilité
    
    def run_parallel_optimizations(self, configs: List[OptimizationConfig], 
                                 obj_fun: Callable, dim: int, 
                                 lb: np.ndarray, ub: np.ndarray,
                                 function_name: str) -> List[OptimizationResult]:
        """Lance plusieurs optimisations en parallèle avec ThreadPoolExecutor"""
        
        def run_single_optimization(config):
            """Fonction d'optimisation unique pour le threading"""
            try:
                if config.algorithm == "SMA":
                    optimizer = EnhancedSMAOptimizer(config)
                elif config.algorithm == "PSO":
                    optimizer = PSOOptimizer(config)
                elif config.algorithm == "GA":
                    optimizer = GeneticAlgorithmOptimizer(config)
                else:
                    raise ValueError(f"Unknown algorithm: {config.algorithm}")
                
                return optimizer.optimize(obj_fun, dim, lb, ub, function_name)
            except Exception as e:
                st.error(f"Optimization failed in thread: {e}")
                return None
        
        # Utilisation de ThreadPoolExecutor au lieu de ProcessPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(run_single_optimization, config) for config in configs]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=300)  # Timeout de 5 minutes
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
            
            return results

# -----------------------------
# Implémentation des Algorithmes
# -----------------------------

class BaseOptimizer:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.metrics_history = {
            'diversity': [],
            'exploration_rate': [],
            'best_fitness': []
        }
    
    def _calculate_diversity(self, population: np.ndarray) -> float:
        centroid = np.mean(population, axis=0)
        return np.mean(np.sqrt(np.sum((population - centroid)**2, axis=1)))
    
    def _update_metrics(self, population: np.ndarray, best_fitness: float):
        diversity = self._calculate_diversity(population)
        exploration_rate = diversity / (np.max(population) - np.min(population) + 1e-12)
        
        self.metrics_history['diversity'].append(diversity)
        self.metrics_history['exploration_rate'].append(exploration_rate)
        self.metrics_history['best_fitness'].append(best_fitness)
        
        if 'monitor' in st.session_state:
            st.session_state.monitor.log_diversity(diversity)

class EnhancedSMAOptimizer(BaseOptimizer):
    def _initialize_population(self, dim: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        samples = np.zeros((self.config.population_size, dim))
        for i in range(dim):
            samples[:, i] = self.rng.permutation(self.config.population_size)
        samples = (samples + self.rng.rand(self.config.population_size, dim)) / self.config.population_size
        return lb + (ub - lb) * samples
    
    def _compute_weights(self, fitness: np.ndarray) -> np.ndarray:
        sorted_idx = np.argsort(fitness)
        ranks = np.zeros_like(fitness)
        ranks[sorted_idx] = np.arange(len(fitness))
        
        if fitness[sorted_idx[0]] == fitness[sorted_idx[-1]]:
            return np.ones_like(fitness)
        
        weights = np.exp(-2.0 * ranks / len(fitness))
        return weights ** 1.5
    
    def _adaptive_z_parameter(self, iteration: int, diversity: float) -> float:
        if not self.config.adaptive_z:
            return self.config.z_param
        
        progress = iteration / self.config.max_iter
        diversity_factor = 1.0 - (diversity / self.metrics_history['diversity'][0] if self.metrics_history['diversity'] else 1.0)
        
        return self.config.z_param * (1.0 - progress**1.5) * (0.8 + 0.2 * diversity_factor)
    
    def _restart_mechanism(self, population: np.ndarray, fitness: np.ndarray, 
                          lb: np.ndarray, ub: np.ndarray, iteration: int):
        diversity = self._calculate_diversity(population)
        
        if diversity < 0.005 and iteration > self.config.max_iter * 0.3:
            restart_count = int(len(population) * 0.3)
            restart_indices = np.argsort(fitness)[-restart_count:]
            
            for idx in restart_indices:
                population[idx] = lb + (ub - lb) * self.rng.rand(len(lb))
                fitness[idx] = self._evaluate_individual(population[idx])
    
    def _evaluate_individual(self, individual: np.ndarray) -> float:
        if 'monitor' in st.session_state:
            st.session_state.monitor.log_function_evaluation()
        
        cache_key = tuple(individual)
        if cache_key in function_cache:
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_cache_hit()
            return function_cache[cache_key]
        else:
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_cache_miss()
            result = self.obj_fun(individual)
            function_cache[cache_key] = result
            return result
    
    def optimize(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray, 
                 function_name: str = "Unknown") -> OptimizationResult:
        start_time = time.perf_counter()
        self.obj_fun = obj_fun
        
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        population = self._initialize_population(dim, lb, ub)
        
        fitness = np.array([self._evaluate_individual(ind) for ind in population])
        
        if not self.config.minimization:
            fitness = -fitness
            
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        history = [best_fitness if self.config.minimization else -best_fitness]
        stagnation_count = 0
        convergence_iter = 0
        
        for iteration in range(self.config.max_iter):
            if iteration % 50 == 0 and iteration > 0:
                self._restart_mechanism(population, fitness, lb, ub, iteration)
            
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
            
            self._update_metrics(population, best_fitness)
            
            if stagnation_count >= self.config.early_stop:
                if self.metrics_history['diversity'][-1] < 0.01:
                    break
            
            current_diversity = self.metrics_history['diversity'][-1] if self.metrics_history['diversity'] else 1.0
            z = self._adaptive_z_parameter(iteration, current_diversity)
            weights = self._compute_weights(fitness)
            
            new_population = self._update_positions(population, best_position, weights, z, lb, ub)
            
            new_fitness = np.array([self._evaluate_individual(ind) for ind in new_population])
            
            if not self.config.minimization:
                new_fitness = -new_fitness
                
            improve_mask = new_fitness < fitness
            population[improve_mask] = new_population[improve_mask]
            fitness[improve_mask] = new_fitness[improve_mask]
            
            if fitness[0] > best_fitness:
                population[0] = best_position
                fitness[0] = best_fitness
            
        execution_time = time.perf_counter() - start_time
        
        if 'monitor' in st.session_state:
            st.session_state.monitor.log_execution_time(execution_time)
        
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
        pop_size, dim = population.shape
        
        r1 = self.rng.rand(pop_size, dim)
        r2 = self.rng.rand(pop_size, dim)
        
        partner_indices = self.rng.randint(0, pop_size, size=pop_size)
        partners = population[partner_indices]
        
        if self.rng.rand() < 0.1:
            levy_step = self._generate_levy_flight(pop_size, dim)
            levy_component = 0.1 * levy_step
        else:
            levy_component = 0
        
        move_toward_best = weights[:, None] * (best_position - population) * r1
        move_toward_partner = (partners - population) * r2
        
        new_positions = population + z * move_toward_best + (1 - z) * move_toward_partner + levy_component
        
        return np.clip(new_positions, lb, ub)
    
    def _generate_levy_flight(self, size: int, dim: int) -> np.ndarray:
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        
        u = self.rng.normal(0, sigma, size=(size, dim))
        v = self.rng.normal(0, 1, size=(size, dim))
        step = u / (np.abs(v) ** (1 / beta))
        
        return step

class PSOOptimizer(BaseOptimizer):
    def optimize(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray, 
                 function_name: str = "Unknown") -> OptimizationResult:
        start_time = time.perf_counter()
        
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        
        population = lb + (ub - lb) * self.rng.rand(self.config.population_size, dim)
        velocity = np.zeros_like(population)
        
        fitness = np.array([obj_fun(ind) for ind in population])
        personal_best_pos = population.copy()
        personal_best_fitness = fitness.copy()
        
        if not self.config.minimization:
            fitness = -fitness
            personal_best_fitness = -personal_best_fitness
            
        best_idx = np.argmin(personal_best_fitness)
        best_position = personal_best_pos[best_idx].copy()
        best_fitness = personal_best_fitness[best_idx]
        
        history = [best_fitness if self.config.minimization else -best_fitness]
        
        for iteration in range(self.config.max_iter):
            r1 = self.rng.rand(self.config.population_size, dim)
            r2 = self.rng.rand(self.config.population_size, dim)
            
            velocity = (self.config.w * velocity + 
                       self.config.c1 * r1 * (personal_best_pos - population) + 
                       self.config.c2 * r2 * (best_position - population))
            
            population = np.clip(population + velocity, lb, ub)
            
            fitness = np.array([obj_fun(ind) for ind in population])
            if not self.config.minimization:
                fitness = -fitness
            
            improved_mask = fitness < personal_best_fitness
            personal_best_pos[improved_mask] = population[improved_mask]
            personal_best_fitness[improved_mask] = fitness[improved_mask]
            
            current_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[current_best_idx] < best_fitness:
                best_fitness = personal_best_fitness[current_best_idx]
                best_position = personal_best_pos[current_best_idx].copy()
            
            history.append(best_fitness if self.config.minimization else -best_fitness)
            self._update_metrics(population, best_fitness)
            
        execution_time = time.perf_counter() - start_time
        
        if 'monitor' in st.session_state:
            st.session_state.monitor.log_execution_time(execution_time)
        
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
    def optimize(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray, 
                 function_name: str = "Unknown") -> OptimizationResult:
        start_time = time.perf_counter()
        
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        
        population = lb + (ub - lb) * self.rng.rand(self.config.population_size, dim)
        fitness = np.array([obj_fun(ind) for ind in population])
        
        if not self.config.minimization:
            fitness = -fitness
            
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        history = [best_fitness if self.config.minimization else -best_fitness]
        
        for iteration in range(self.config.max_iter):
            parents = self._tournament_selection(population, fitness)
            
            offspring = self._crossover(parents)
            
            offspring = self._mutate(offspring, lb, ub)
            
            offspring_fitness = np.array([obj_fun(ind) for ind in offspring])
            if not self.config.minimization:
                offspring_fitness = -offspring_fitness
            
            combined_population = np.vstack([population, offspring])
            combined_fitness = np.hstack([fitness, offspring_fitness])
            
            best_indices = np.argsort(combined_fitness)[:self.config.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]
            
            if fitness[0] < best_fitness:
                best_fitness = fitness[0]
                best_position = population[0].copy()
            
            history.append(best_fitness if self.config.minimization else -best_fitness)
            self._update_metrics(population, best_fitness)
            
        execution_time = time.perf_counter() - start_time
        
        if 'monitor' in st.session_state:
            st.session_state.monitor.log_execution_time(execution_time)
        
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
        selected = []
        for _ in range(self.config.population_size):
            indices = self.rng.choice(len(population), size=3, replace=False)
            best_idx = indices[np.argmin(fitness[indices])]
            selected.append(population[best_idx])
        return np.array(selected)
    
    def _crossover(self, parents: np.ndarray) -> np.ndarray:
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
        mutated = population.copy()
        for i in range(len(population)):
            if self.rng.rand() < self.config.mutation_rate:
                for j in range(population.shape[1]):
                    u = self.rng.rand()
                    delta = (2 * u) ** (1 / (1 + 20)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (1 + 20))
                    mutated[i, j] += delta * (ub[j] - lb[j])
        return np.clip(mutated, lb, ub)

ALGORITHMS = {
    'SMA': EnhancedSMAOptimizer,
    'PSO': PSOOptimizer,
    'GA': GeneticAlgorithmOptimizer,
}

# -----------------------------
# Hyperparameter Optimization
# -----------------------------

class HyperparameterOptimizer:
    def __init__(self):
        self.param_space = {
            'population_size': (20, 200),
            'z_param': (0.01, 0.1),
            'early_stop': (50, 500)
        }
    
    def optimize_hyperparameters(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray,
                                n_trials: int = 20) -> Dict[str, Any]:
        best_score = float('inf')
        best_params = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for trial in range(n_trials):
            status_text.text(f"Hyperparameter trial {trial + 1}/{n_trials}")
            progress_bar.progress((trial + 1) / n_trials)
            
            params = {
                'population_size': np.random.randint(20, 200),
                'z_param': np.random.uniform(0.01, 0.1),
                'early_stop': np.random.randint(50, 500)
            }
            
            config = OptimizationConfig(**params)
            optimizer = EnhancedSMAOptimizer(config)
            result = optimizer.optimize(obj_fun, dim, lb, ub)
            
            if result.best_fitness < best_score:
                best_score = result.best_fitness
                best_params = params
        
        progress_bar.empty()
        status_text.empty()
        
        return {'best_params': best_params, 'best_score': best_score}

# -----------------------------
# Project Management System
# -----------------------------

class ProjectManager:
    def __init__(self):
        self.projects = {}
        self.current_project = None
    
    def create_project(self, name: str, description: str = ""):
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
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Convergence History', 'Population Diversity', 
                       'Exploration vs Exploitation', 'Algorithm Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for i, result in enumerate(results):
        fig.add_trace(
            go.Scatter(y=result.history, name=f'{result.algorithm} Run {i+1}',
                      line=dict(width=2)),
            row=1, col=1
        )
    
    for i, result in enumerate(results):
        if 'metrics_history' in result.metadata:
            diversity = result.metadata['metrics_history']['diversity']
            fig.add_trace(
                go.Scatter(y=diversity, name=f'{result.algorithm} Diversity',
                          line=dict(dash='dot')),
                row=1, col=2
            )
    
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

def create_3d_visualization(results: List[OptimizationResult], params: dict):
    if params['dim'] < 2:
        st.warning("3D visualization requires at least 2 dimensions")
        return None
    
    try:
        benchmark = BENCHMARKS[params['func_name']]
        bounds = benchmark['bounds']
        
        x = np.linspace(bounds[0], bounds[1], 30)
        y = np.linspace(bounds[0], bounds[1], 30)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i,j], Y[i,j]])
                full_point = np.full(params['dim'], point[0])
                full_point[:2] = point
                Z[i,j] = benchmark['func'](full_point)
        
        fig = go.Figure(data=[
            go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.7,
                      contours=dict(z=dict(show=True, usecolormap=True, project_z=True))),
        ])
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, result in enumerate(results):
            if len(result.best_position) >= 2:
                fig.add_trace(go.Scatter3d(
                    x=[result.best_position[0]],
                    y=[result.best_position[1]],
                    z=[result.best_fitness],
                    mode='markers',
                    marker=dict(size=8, color=colors[idx % len(colors)]),
                    name=f'{result.algorithm} Best'
                ))
        
        fig.update_layout(
            title=f"3D Landscape: {params['func_name']}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Fitness'
            ),
            height=500
        )
        
        return fig
    except Exception as e:
        st.warning(f"3D visualization not available: {e}")
        return None

def create_parallel_coordinates_plot(results: List[OptimizationResult]):
    try:
        data = []
        for result in results:
            data.append({
                'Algorithm': result.algorithm,
                'Best Fitness': result.best_fitness,
                'Execution Time': result.execution_time,
                'Iterations': result.iterations,
                'Convergence Iter': result.convergence_iter,
                'Dimension': result.dimension
            })
        
        df = pd.DataFrame(data)
        fig = px.parallel_coordinates(
            df,
            color='Best Fitness',
            dimensions=['Best Fitness', 'Execution Time', 'Iterations', 'Convergence Iter'],
            color_continuous_scale=px.colors.diverging.Tealrose,
            title="Parallel Coordinates Analysis"
        )
        
        return fig
    except Exception as e:
        st.warning(f"Parallel coordinates plot not available: {e}")
        return None

# -----------------------------
# Statistical Analysis
# -----------------------------

class StatisticalAnalyzer:
    @staticmethod
    def wilcoxon_signed_rank_test(results_a: List[OptimizationResult], 
                                 results_b: List[OptimizationResult]) -> Dict[str, Any]:
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
        return {
            'auc': np.trapz(history),
            'final_improvement': history[0] - history[-1],
            'convergence_speed': len(history) / (history[0] - history[-1] + 1e-12),
            'stability': np.std(history[-10:]) if len(history) >= 10 else np.std(history)
        }

# -----------------------------
# Enhanced Export Utilities
# -----------------------------

def create_comprehensive_report(results: List[OptimizationResult], 
                               project_info: Dict[str, Any]) -> Dict[str, Any]:
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
    
    for algo in set([r.algorithm for r in results]):
        algo_results = [r for r in results if r.algorithm == algo]
        report['detailed_results'][algo] = {
            'best_fitness': min([r.best_fitness for r in algo_results]),
            'worst_fitness': max([r.best_fitness for r in algo_results]),
            'average_fitness': np.mean([r.best_fitness for r in algo_results]),
            'std_fitness': np.std([r.best_fitness for r in algo_results]),
            'success_rate': np.mean([1 if abs(r.best_fitness - BENCHMARKS[r.function_name]['global_min']) < 1e-6 else 0 for r in algo_results])
        }
    
    best_algo = min(report['detailed_results'].items(), 
                   key=lambda x: x[1]['average_fitness'])[0]
    report['recommendations'].append(f"Best performing algorithm: {best_algo}")
    
    return report

# -----------------------------
# Streamlit UI - Enhanced
# -----------------------------

def create_advanced_controls():
    with st.sidebar.expander("⚙️ Advanced Controls", expanded=False):
        use_parallel = st.checkbox("Enable Parallel Processing", value=False)  # Désactivé par défaut
        max_workers = st.slider("Max Workers", 1, min(4, cpu_count()), 2)  # Limité à 2
        
        enable_caching = st.checkbox("Enable Result Caching", value=True)
        cache_size = st.slider("Cache Size", 10, 200, 50)
        
        use_adaptive_params = st.checkbox("Adaptive Parameters", value=True)
        diversity_threshold = st.slider("Diversity Threshold", 0.001, 0.1, 0.01, 0.001)
        
        enable_3d = st.checkbox("Enable 3D Visualization", value=True)
        enable_parallel_plot = st.checkbox("Enable Parallel Coordinates", value=True)
        
        return {
            'use_parallel': use_parallel,
            'max_workers': max_workers,
            'enable_caching': enable_caching,
            'cache_size': cache_size,
            'use_adaptive_params': use_adaptive_params,
            'diversity_threshold': diversity_threshold,
            'enable_3d': enable_3d,
            'enable_parallel_plot': enable_parallel_plot
        }

def setup_sidebar():
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
            dim = st.slider("Dimension", 1, 50, 10)  # Réduit la dimension max
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
            pop_size = st.slider("Population", 10, 200, 30)  # Réduit la population max
            max_iter = st.slider("Max Iterations", 50, 1000, 200)  # Réduit les itérations
        with col2:
            num_runs = st.slider("Runs per Algorithm", 1, 10, 2)  # Réduit le nombre de runs
            seed = st.number_input("Seed", value=42)
        
        if user_level == "Expert":
            st.subheader("Advanced Parameters")
            early_stop = st.slider("Early Stop", 10, 500, 50)
            adaptive_z = st.checkbox("Adaptive Z", True)
            z_param = st.slider("Z Parameter", 0.001, 0.1, 0.03, 0.001)
        else:
            early_stop = 100
            adaptive_z = True
            z_param = 0.03
        
        if user_level == "Expert":
            optimize_hyperparams = st.checkbox("Optimize Hyperparameters", False)
        else:
            optimize_hyperparams = False
        
        submitted = st.form_submit_button("🎯 Run Optimization")
    
    advanced_controls = {}
    if user_level == "Expert":
        advanced_controls = create_advanced_controls()
    
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
        'submitted': submitted,
        'advanced_controls': advanced_controls
    }

def display_enhanced_results(results: List[OptimizationResult], params: dict):
    if not results:
        st.error("No results to display - all optimizations failed")
        return
        
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
        success_rate = np.mean([1 if abs(r.best_fitness - BENCHMARKS[r.function_name]['global_min']) < 1e-6 else 0 for r in results])
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    if 'monitor' in st.session_state:
        st.subheader("📈 Performance Monitoring")
        st.session_state.monitor.display_metrics()
    
    if PLOTLY_AVAILABLE and results:
        st.subheader("📈 Live Dashboard")
        dashboard = create_real_time_dashboard(results)
        st.plotly_chart(dashboard, use_container_width=True)
    
    if results:
        st.subheader("🔄 Algorithm Comparison")
        comp_fig = plot_algorithm_comparison(results)
        st.plotly_chart(comp_fig, use_container_width=True)
    
    if params.get('advanced_controls', {}).get('enable_3d', False) and results:
        st.subheader("🌐 3D Landscape Visualization")
        fig_3d = create_3d_visualization(results, params)
        if fig_3d:
            st.plotly_chart(fig_3d, use_container_width=True)
    
    if params.get('advanced_controls', {}).get('enable_parallel_plot', False) and results:
        st.subheader("📊 Parallel Coordinates Analysis")
        parallel_fig = create_parallel_coordinates_plot(results)
        if parallel_fig:
            st.plotly_chart(parallel_fig, use_container_width=True)
    
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
    
    if len(params['algorithms']) > 1 and results:
        st.subheader("📊 Statistical Analysis")
        analyzer = StatisticalAnalyzer()
        
        algorithms = list(set([r.algorithm for r in results]))
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                algo_a_results = [r for r in results if r.algorithm == algorithms[i]]
                algo_b_results = [r for r in results if r.algorithm == algorithms[j]]
                
                test_result = analyzer.wilcoxon_signed_rank_test(algo_a_results, algo_b_results)
                
                if 'error' not in test_result:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{algorithms[i]} vs {algorithms[j]}**")
                    with col2:
                        st.write(f"p-value: {test_result['p_value']:.4f}")
                    with col3:
                        if test_result['significant']:
                            st.success("Significant difference")
                        else:
                            st.warning("No significant difference")

def setup_advanced_export(project_manager: ProjectManager, results: List[OptimizationResult], params: dict):
    st.sidebar.header("💾 Advanced Export")
    
    if st.sidebar.button("💼 Save Project") and results:
        project_id = project_manager.create_project(
            f"Optimization_{params['func_name']}",
            f"Multi-algorithm comparison on {params['func_name']}"
        )
        session_id = project_manager.save_session(
            OptimizationConfig(**{k: v for k, v in params.items() if k in OptimizationConfig.__annotations__}),
            results
        )
        st.sidebar.success(f"Project saved: {project_id}")
    
    if st.sidebar.button("📄 Generate Full Report") and results:
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
    
    if st.sidebar.button("📊 Export Comparison Data") and results:
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
    
    *Note: Parallel processing is disabled by default for stability.*
    """)
    
    if 'project_manager' not in st.session_state:
        st.session_state.project_manager = ProjectManager()
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = []
    if 'monitor' not in st.session_state:
        st.session_state.monitor = RealTimeMonitor()
    if 'results_cache' not in st.session_state:
        st.session_state.results_cache = ResultsCache()
    if 'logger' not in st.session_state:
        st.session_state.logger = OptimizationLogger()
    
    with st.expander("🔧 System Information", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write(f"**Numba:** {NUMBA_AVAILABLE}")
        with col2:
            st.write(f"**Plotly:** {PLOTLY_AVAILABLE}")
        with col3:
            st.write(f"**Algorithms:** {len(ALGORITHMS)}")
        with col4:
            st.write(f"**Benchmarks:** {len(BENCHMARKS)}")
        with col5:
            st.write(f"**CPU Cores:** {cpu_count()}")
    
    params = setup_sidebar()
    
    if params['submitted']:
        benchmark = BENCHMARKS[params['func_name']]
        obj_fun = benchmark['func']
        bounds = benchmark['bounds']
        
        all_results = []
        
        if params['optimize_hyperparams'] and 'SMA' in params['algorithms']:
            with st.spinner("🔍 Optimizing SMA hyperparameters..."):
                hyper_optimizer = HyperparameterOptimizer()
                hyper_result = hyper_optimizer.optimize_hyperparameters(
                    obj_fun, params['dim'], 
                    np.full(params['dim'], bounds[0]),
                    np.full(params['dim'], bounds[1])
                )
                st.success(f"Best parameters: {hyper_result['best_params']}")
        
        configs = []
        for algorithm in params['algorithms']:
            for run in range(params['num_runs']):
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
                configs.append(config)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if params.get('advanced_controls', {}).get('use_parallel', False):
            status_text.text("🚀 Running parallel optimizations...")
            max_workers = params['advanced_controls']['max_workers']
            parallel_optimizer = ParallelOptimizer(max_workers=max_workers)
            
            all_results = parallel_optimizer.run_parallel_optimizations(
                configs, obj_fun, params['dim'],
                np.full(params['dim'], bounds[0]),
                np.full(params['dim'], bounds[1]),
                params['func_name']
            )
            
            progress_bar.progress(1.0)
        else:
            # Exécution séquentielle - plus stable
            total_tasks = len(configs)
            completed_tasks = 0
            
            for config in configs:
                status_text.text(f"Running {config.algorithm} - Run {completed_tasks + 1}/{total_tasks}")
                
                try:
                    optimizer_class = ALGORITHMS[config.algorithm]
                    optimizer = optimizer_class(config)
                    
                    lb = np.full(params['dim'], bounds[0])
                    ub = np.full(params['dim'], bounds[1])
                    
                    result = optimizer.optimize(obj_fun, params['dim'], lb, ub, params['func_name'])
                    all_results.append(result)
                    
                except Exception as e:
                    st.error(f"Optimization failed for {config.algorithm}: {e}")
                
                completed_tasks += 1
                progress_bar.progress(completed_tasks / total_tasks)
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.optimization_history.extend(all_results)
        
        display_enhanced_results(all_results, params)
        
        setup_advanced_export(st.session_state.project_manager, all_results, params)
        
        if all_results:
            st.sidebar.header("🎯 Performance Insights")
            best_algo = min(set([r.algorithm for r in all_results]), 
                           key=lambda algo: np.mean([r.best_fitness for r in all_results if r.algorithm == algo]))
            st.sidebar.success(f"**Recommended Algorithm:** {best_algo}")
            
            avg_improvement = np.mean([r.history[0] - r.history[-1] for r in all_results if len(r.history) > 1])
            st.sidebar.info(f"**Average Improvement:** {avg_improvement:.2e}")
            
            st.session_state.logger.log_optimization_end(all_results[0])
        else:
            st.sidebar.error("❌ No successful optimizations")

if __name__ == "__main__":
    main()
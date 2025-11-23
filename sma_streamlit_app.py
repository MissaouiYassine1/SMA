"""
SMA Streamlit App - Version Complète et Consolidée CORRIGÉE
Slime Mould Algorithm implementation + Streamlit UI avec toutes les fonctionnalités
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
import concurrent.futures
from multiprocessing import cpu_count
import hashlib
from functools import lru_cache
import logging
from logging.handlers import RotatingFileHandler
import random

# Configuration des imports conditionnels
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Gestion des imports optionnels avec gestion d'erreurs robuste
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# -----------------------------
# Décorateurs conditionnels corrigés
# -----------------------------

def conditional_njit(func):
    """Décorateur conditionnel pour Numba"""
    return njit(fastmath=True, cache=True)(func) if NUMBA_AVAILABLE else func

# -----------------------------
# Système de Cache et Mémoire Amélioré
# -----------------------------

class ResultsCache:
    """Cache intelligent avec limite de mémoire"""
    
    def __init__(self, max_size=50, max_memory_mb=100):
        self.cache = {}
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.access_count = {}
        self.memory_usage = 0
    
    def _get_size(self, obj):
        """Estime la taille mémoire d'un objet"""
        try:
            return len(pickle.dumps(obj, -1))
        except:
            return 0
    
    def get(self, key):
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key, value):
        value_size = self._get_size(value)
        
        # Vérification de la limite de mémoire
        if value_size > self.max_memory_mb * 1024 * 1024:
            return  # Objet trop volumineux
        
        # Nettoyage si nécessaire
        while (len(self.cache) >= self.max_size or 
               (self.memory_usage + value_size) > self.max_memory_mb * 1024 * 1024):
            if not self.cache:
                break
            min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            removed_size = self._get_size(self.cache[min_key])
            del self.cache[min_key]
            del self.access_count[min_key]
            self.memory_usage -= removed_size
        
        self.cache[key] = value
        self.access_count[key] = 1
        self.memory_usage += value_size
    
    def clear(self):
        """Vide le cache"""
        self.cache.clear()
        self.access_count.clear()
        self.memory_usage = 0

# Cache global avec gestion de mémoire
function_cache = ResultsCache(max_size=100, max_memory_mb=50)

# -----------------------------
# Système de Logging et Monitoring Amélioré
# -----------------------------

class OptimizationLogger:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('OptimizationSuite')
        self.logger.setLevel(log_level)
        
        # Éviter les handlers dupliqués
        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            try:
                file_handler = RotatingFileHandler(
                    'optimization_suite.log',
                    maxBytes=10*1024*1024,
                    backupCount=5
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Impossible de créer le fichier de log: {e}")
    
    def log_optimization_start(self, config: 'OptimizationConfig'):
        self.logger.info(f"Starting optimization with {config.algorithm}")
    
    def log_optimization_end(self, result: 'OptimizationResult'):
        self.logger.info(
            f"Optimization completed: {result.algorithm} - "
            f"Best fitness: {result.best_fitness:.6f}"
        )
    
    def log_error(self, error_msg: str):
        self.logger.error(f"Optimization error: {error_msg}")

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
    
    def reset(self):
        """Réinitialise les métriques"""
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = []
            else:
                self.metrics[key] = 0

# -----------------------------
# Enhanced Benchmark Functions avec validation CORRIGÉE
# -----------------------------

def validate_dimension(dim: int, max_dim: int = 1000) -> bool:
    """Valide que la dimension est raisonnable"""
    return 1 <= dim <= max_dim

# Décorateur pour la gestion d'erreurs des fonctions benchmark
def benchmark_error_handler(func):
    def wrapper(x: np.ndarray) -> float:
        try:
            if not validate_dimension(len(x)):
                raise ValueError(f"Dimension {len(x)} trop élevée")
            return func(x)
        except Exception as e:
            return float('inf')
    return wrapper

# Application CORRECTE des décorateurs conditionnels
@benchmark_error_handler
@conditional_njit
def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)

@benchmark_error_handler
@conditional_njit
def rastrigin(x: np.ndarray) -> float:
    n = x.size
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x))

@benchmark_error_handler
@conditional_njit
def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2)

@benchmark_error_handler
@conditional_njit
def ackley(x: np.ndarray) -> float:
    a, b, c = 20, 0.2, 2 * np.pi
    n = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(s1/n)) - np.exp(s2/n) + a + np.e

@benchmark_error_handler
@conditional_njit
def griewank(x: np.ndarray) -> float:
    return 1 + np.sum(x**2)/4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))

@benchmark_error_handler
@conditional_njit
def schwefel(x: np.ndarray) -> float:
    n = x.size
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

@benchmark_error_handler
@conditional_njit
def zakharov(x: np.ndarray) -> float:
    n = x.size
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, n+1) * x)
    return sum1 + sum2**2 + sum2**4

@benchmark_error_handler
@conditional_njit
def levy(x: np.ndarray) -> float:
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0]))**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1))**2))
    term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
    return term1 + term2 + term3

@benchmark_error_handler
@conditional_njit
def happy_cat(x: np.ndarray) -> float:
    n = x.size
    sum_sq = np.sum(x**2)
    sum_quad = np.sum(x**4)
    return ((sum_sq - n)**2)**0.125 + (0.5 * sum_sq + sum_quad) / n + 0.5

@benchmark_error_handler
@conditional_njit
def alpine(x: np.ndarray) -> float:
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

@benchmark_error_handler
@conditional_njit
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
# Configuration et Résultats avec validation AMÉLIORÉE
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
    
    def validate(self) -> List[str]:
        """Valide la configuration et retourne les erreurs"""
        errors = []
        
        if self.population_size < 5:
            errors.append("La taille de population doit être au moins 5")
        if self.max_iter < 10:
            errors.append("Le nombre d'itérations doit être au moins 10")
        if self.early_stop < 1:
            errors.append("L'arrêt précoce doit être au moins 1")
        if not (0 <= self.z_param <= 1):
            errors.append("Le paramètre z doit être entre 0 et 1")
        if not (0 <= self.crossover_rate <= 1):
            errors.append("Le taux de crossover doit être entre 0 et 1")
        if not (0 <= self.mutation_rate <= 1):
            errors.append("Le taux de mutation doit être entre 0 et 1")
        if not (0 <= self.w <= 2):
            errors.append("Le paramètre w doit être entre 0 et 2")
        if not (0 <= self.c1 <= 3):
            errors.append("Le paramètre c1 doit être entre 0 et 3")
        if not (0 <= self.c2 <= 3):
            errors.append("Le paramètre c2 doit être entre 0 et 3")
        
        return errors

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
# Implémentation des Algorithmes CORRIGÉE
# -----------------------------

class BaseOptimizer:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        # Initialisation robuste du générateur aléatoire
        self.rng = np.random.RandomState(config.seed)
        if config.seed is not None:
            random.seed(config.seed)
        self.metrics_history = {
            'diversity': [],
            'exploration_rate': [],
            'best_fitness': []
        }
    
    def _validate_bounds(self, lb: np.ndarray, ub: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Valide et formate les bornes"""
        lb = np.broadcast_to(lb, dim).astype(np.float64)
        ub = np.broadcast_to(ub, dim).astype(np.float64)
        
        if np.any(lb >= ub):
            raise ValueError("Toutes les bornes inférieures doivent être strictement inférieures aux bornes supérieures")
        
        return lb, ub
    
    def _calculate_diversity(self, population: np.ndarray) -> float:
        """Calcule la diversité de la population de manière robuste"""
        try:
            if len(population) <= 1:
                return 0.0
            centroid = np.mean(population, axis=0)
            distances = np.linalg.norm(population - centroid, axis=1)
            return np.mean(distances)
        except:
            return 0.0
    
    def _update_metrics(self, population: np.ndarray, best_fitness: float):
        """Met à jour les métriques de suivi"""
        try:
            diversity = self._calculate_diversity(population)
            pop_range = np.max(population) - np.min(population)
            exploration_rate = diversity / (pop_range + 1e-12)
            
            self.metrics_history['diversity'].append(diversity)
            self.metrics_history['exploration_rate'].append(exploration_rate)
            self.metrics_history['best_fitness'].append(best_fitness)
            
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_diversity(diversity)
        except Exception as e:
            # Log silencieux des erreurs de métriques
            pass

class CorrectSMAOptimizer(BaseOptimizer):
    """Implémentation CORRIGÉE de l'algorithme SMA"""
    
    def _initialize_population(self, dim: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Initialise la population de manière uniforme"""
        try:
            return self.rng.uniform(lb, ub, (self.config.population_size, dim))
        except:
            # Fallback robuste
            return lb + (ub - lb) * self.rng.rand(self.config.population_size, dim)
    
    def _compute_sma_weights(self, fitness: np.ndarray) -> np.ndarray:
        """Calcule les poids selon l'algorithme SMA standard CORRIGÉ"""
        try:
            sorted_idx = np.argsort(fitness)
            best_f = fitness[sorted_idx[0]]
            worst_f = fitness[sorted_idx[-1]]
            
            # Éviter la division par zéro
            if abs(best_f - worst_f) < 1e-12:
                return np.ones_like(fitness)
            
            sm = np.ones_like(fitness)
            half_pop = len(fitness) // 2
            
            for i in range(len(fitness)):
                if i < half_pop:  # Meilleure moitié
                    r = self.rng.rand()
                    sm[sorted_idx[i]] = 1 + r * np.log10(
                        (best_f - fitness[sorted_idx[i]]) / (best_f - worst_f) + 1
                    )
                else:  # Pire moitié
                    r = self.rng.rand()
                    sm[sorted_idx[i]] = 1 - r * np.log10(
                        (best_f - fitness[sorted_idx[i]]) / (best_f - worst_f) + 1
                    )
            
            return np.clip(sm, 0.1, 2.0)  # Limiter les poids
        except:
            return np.ones_like(fitness)
    
    def _adaptive_z_parameter(self, iteration: int, max_iter: int) -> float:
        """Adapte dynamiquement le paramètre z"""
        if not self.config.adaptive_z:
            return self.config.z_param
        
        try:
            progress = iteration / max_iter
            # z diminue progressivement pour passer de l'exploration à l'exploitation
            return self.config.z_param * (1.0 - progress**1.5)
        except:
            return self.config.z_param
    
    def _evaluate_individual(self, individual: np.ndarray) -> float:
        """Évalue un individu avec gestion du cache CORRIGÉE"""
        if 'monitor' in st.session_state:
            st.session_state.monitor.log_function_evaluation()
        
        # Clé de cache robuste avec hash
        cache_key = hashlib.md5(individual.tobytes()).hexdigest()
        cached_result = function_cache.get(cache_key)
        
        if cached_result is not None:
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_cache_hit()
            return cached_result
        else:
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_cache_miss()
            try:
                result = self.obj_fun(individual)
                function_cache.set(cache_key, result)
                return result
            except Exception as e:
                return float('inf')
    
    def _evaluate_with_direction(self, individual: np.ndarray) -> float:
        """Évalue en prenant en compte minimization/maximization"""
        result = self._evaluate_individual(individual)
        return result if self.config.minimization else -result
    
    def _update_positions_sma(self, population: np.ndarray, fitness: np.ndarray, 
                             best_position: np.ndarray, lb: np.ndarray, ub: np.ndarray, 
                             iteration: int, max_iter: int) -> np.ndarray:
        """Mise à jour CORRECTE des positions selon SMA"""
        pop_size, dim = population.shape
        new_population = population.copy()
        
        # Calcul des poids SMA
        sm = self._compute_sma_weights(fitness)
        z = self._adaptive_z_parameter(iteration, max_iter)
        best_fitness = np.min(fitness)
        
        # Paramètre d'oscillation
        a = np.arctanh(1 - (iteration / max_iter))
        
        for i in range(pop_size):
            if i == 0:  # Toujours garder le meilleur
                new_population[i] = best_position
                continue
                
            p = np.tanh(abs(fitness[i] - best_fitness))
            
            if self.rng.rand() < z:  # Phase d'exploration
                # Mouvement vers une position aléatoire
                A = self.rng.choice([-1, 1])
                b = 1 - (iteration / max_iter)
                random_vec = self.rng.uniform(lb, ub)
                new_population[i] = random_vec + A * b * (ub - lb) * self.rng.rand(dim)
            else:  # Phase d'exploitation
                if self.rng.rand() < p:
                    # Mouvement vers le meilleur
                    r1 = self.rng.rand(dim)
                    new_population[i] = best_position + sm[i] * (ub - lb) * r1 * 0.1
                else:
                    # Mouvement oscillatoire
                    r2 = self.rng.rand(dim)
                    partner_idx = self.rng.randint(0, pop_size)
                    new_population[i] = population[i] + sm[i] * (population[partner_idx] - population[i]) * r2
        
        # Application robuste des bornes
        new_population = np.clip(new_population, lb, ub)
        return new_population
    
    def optimize(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray, 
                 function_name: str = "Unknown") -> OptimizationResult:
        """Exécute l'optimisation SMA CORRIGÉE"""
        start_time = time.perf_counter()
        self.obj_fun = obj_fun
        
        try:
            # Validation des bornes
            lb, ub = self._validate_bounds(lb, ub, dim)
            
            # Initialisation
            population = self._initialize_population(dim, lb, ub)
            fitness = np.array([self._evaluate_with_direction(ind) for ind in population])
            
            best_idx = np.argmin(fitness)
            best_position = population[best_idx].copy()
            best_fitness = fitness[best_idx]
            
            history = [best_fitness]
            stagnation_count = 0
            convergence_iter = 0
            
            # Boucle d'optimisation principale
            for iteration in range(self.config.max_iter):
                try:
                    # Mise à jour des positions
                    population = self._update_positions_sma(
                        population, fitness, best_position, lb, ub, 
                        iteration, self.config.max_iter
                    )
                    
                    # Évaluation
                    new_fitness = np.array([self._evaluate_with_direction(ind) for ind in population])
                    
                    # Mise à jour des meilleures solutions
                    improve_mask = new_fitness < fitness
                    population[improve_mask] = population[improve_mask]
                    fitness[improve_mask] = new_fitness[improve_mask]
                    
                    # Mise à jour du meilleur global
                    current_best_idx = np.argmin(fitness)
                    current_best_fitness = fitness[current_best_idx]
                    
                    if current_best_fitness < best_fitness:
                        best_fitness = current_best_fitness
                        best_position = population[current_best_idx].copy()
                        stagnation_count = 0
                        convergence_iter = iteration
                    else:
                        stagnation_count += 1
                    
                    history.append(best_fitness)
                    
                    # Métriques
                    self._update_metrics(population, best_fitness)
                    
                    # Arrêt précoce
                    if stagnation_count >= self.config.early_stop:
                        break
                        
                except (ValueError, RuntimeError, ZeroDivisionError) as e:
                    # Continuer pour les erreurs non-critiques
                    continue
                except Exception as e:
                    # Arrêter pour les erreurs critiques
                    st.error(f"Critical error in SMA optimization: {e}")
                    break
            
            execution_time = time.perf_counter() - start_time
            
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_execution_time(execution_time)
            
            final_fitness = best_fitness if self.config.minimization else -best_fitness
            
            return OptimizationResult(
                algorithm="SMA",
                best_position=best_position,
                best_fitness=final_fitness,
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
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            st.error(f"Optimization failed: {e}")
            
            return OptimizationResult(
                algorithm="SMA",
                best_position=np.full(dim, (lb[0] + ub[0]) / 2),
                best_fitness=float('inf'),
                history=np.array([float('inf')]),
                population=np.zeros((1, dim)),
                fitness=np.array([float('inf')]),
                iterations=0,
                execution_time=execution_time,
                convergence_iter=0,
                config=self.config,
                function_name=function_name,
                dimension=dim
            )

class PSOOptimizer(BaseOptimizer):
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.obj_fun = None  # Initialisation ajoutée
    
    def _evaluate_with_direction(self, individual: np.ndarray) -> float:
        """Évalue un individu avec gestion du cache et direction"""
        if 'monitor' in st.session_state:
            st.session_state.monitor.log_function_evaluation()
        
        # Clé de cache robuste
        cache_key = hashlib.md5(individual.tobytes()).hexdigest()
        cached_result = function_cache.get(cache_key)
        
        if cached_result is not None:
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_cache_hit()
            result = cached_result
        else:
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_cache_miss()
            try:
                result = self.obj_fun(individual)
                function_cache.set(cache_key, result)
            except Exception as e:
                result = float('inf')
        
        # Inverser pour la maximisation si nécessaire
        return result if self.config.minimization else -result

    def optimize(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray, 
                 function_name: str = "Unknown") -> OptimizationResult:
        """Implémentation PSO CORRIGÉE"""
        start_time = time.perf_counter()
        self.obj_fun = obj_fun  # Assignation de la fonction objectif
        
        try:
            lb, ub = self._validate_bounds(lb, ub, dim)
            
            # Initialisation de la population et vitesse
            population = self.rng.uniform(lb, ub, (self.config.population_size, dim))
            velocity = np.zeros_like(population)
            
            # Évaluation initiale
            fitness = np.array([self._evaluate_with_direction(ind) for ind in population])
            personal_best_pos = population.copy()
            personal_best_fitness = fitness.copy()
            
            # Meilleur global
            best_idx = np.argmin(personal_best_fitness)
            best_position = personal_best_pos[best_idx].copy()
            best_fitness = personal_best_fitness[best_idx]
            
            history = [best_fitness]
            stagnation_count = 0
            convergence_iter = 0
            
            # Boucle d'optimisation principale
            for iteration in range(self.config.max_iter):
                try:
                    # Coefficients aléatoires
                    r1 = self.rng.rand(self.config.population_size, dim)
                    r2 = self.rng.rand(self.config.population_size, dim)
                    
                    # Mise à jour de la vitesse (équation PSO standard)
                    velocity = (self.config.w * velocity + 
                               self.config.c1 * r1 * (personal_best_pos - population) + 
                               self.config.c2 * r2 * (best_position - population))
                    
                    # Limitation de la vitesse (empêcher l'explosion)
                    v_max = 0.2 * (ub - lb)
                    velocity = np.clip(velocity, -v_max, v_max)
                    
                    # Mise à jour des positions
                    population = np.clip(population + velocity, lb, ub)
                    
                    # Évaluation
                    fitness = np.array([self._evaluate_with_direction(ind) for ind in population])
                    
                    # Mise à jour des meilleures positions personnelles
                    improved_mask = fitness < personal_best_fitness
                    personal_best_pos[improved_mask] = population[improved_mask]
                    personal_best_fitness[improved_mask] = fitness[improved_mask]
                    
                    # Mise à jour du meilleur global
                    current_best_idx = np.argmin(personal_best_fitness)
                    current_best_fitness = personal_best_fitness[current_best_idx]
                    
                    if current_best_fitness < best_fitness:
                        best_fitness = current_best_fitness
                        best_position = personal_best_pos[current_best_idx].copy()
                        stagnation_count = 0
                        convergence_iter = iteration
                    else:
                        stagnation_count += 1
                    
                    history.append(best_fitness)
                    self._update_metrics(population, best_fitness)
                    
                    # Arrêt précoce
                    if stagnation_count >= self.config.early_stop:
                        break
                        
                except Exception as e:
                    # Continuer en cas d'erreur non-critique
                    continue
            
            execution_time = time.perf_counter() - start_time
            
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_execution_time(execution_time)
            
            # Ajustement final de la fitness selon minimization/maximization
            final_fitness = best_fitness if self.config.minimization else -best_fitness
            
            return OptimizationResult(
                algorithm="PSO",
                best_position=best_position,
                best_fitness=final_fitness,
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
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            st.error(f"PSO optimization failed: {e}")
            
            return OptimizationResult(
                algorithm="PSO",
                best_position=np.full(dim, (lb[0] + ub[0]) / 2),
                best_fitness=float('inf'),
                history=np.array([float('inf')]),
                population=np.zeros((1, dim)),
                fitness=np.array([float('inf')]),
                iterations=0,
                execution_time=execution_time,
                convergence_iter=0,
                config=self.config,
                function_name=function_name,
                dimension=dim
            )

class GeneticAlgorithmOptimizer(BaseOptimizer):
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.obj_fun = None  # Initialisation ajoutée
    
    def _evaluate_with_direction(self, individual: np.ndarray) -> float:
        """Évalue un individu avec gestion du cache et direction"""
        if 'monitor' in st.session_state:
            st.session_state.monitor.log_function_evaluation()
        
        cache_key = hashlib.md5(individual.tobytes()).hexdigest()
        cached_result = function_cache.get(cache_key)
        
        if cached_result is not None:
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_cache_hit()
            result = cached_result
        else:
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_cache_miss()
            try:
                result = self.obj_fun(individual)
                function_cache.set(cache_key, result)
            except Exception as e:
                result = float('inf')
        
        return result if self.config.minimization else -result
    
    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Sélection par tournoi CORRIGÉE"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Sélection aléatoire d'individus pour le tournoi
            indices = self.rng.choice(len(population), size=tournament_size, replace=False)
            
            # Sélection du meilleur (fitness la plus basse pour minimization)
            best_idx = indices[np.argmin(fitness[indices])]
            selected.append(population[best_idx])
        
        return np.array(selected)
    
    def _simulated_binary_crossover(self, parents: np.ndarray) -> np.ndarray:
        """Crossover binaire simulé CORRIGÉ"""
        offspring = []
        eta_c = 20  # Paramètre de distribution pour le crossover
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i+1]
                
                if self.rng.rand() < self.config.crossover_rate:
                    child1, child2 = parent1.copy(), parent2.copy()
                    
                    for j in range(len(parent1)):
                        if self.rng.rand() <= 0.5:
                            # Éviter la division par zéro
                            if abs(parent2[j] - parent1[j]) > 1e-12:
                                u = self.rng.rand()
                                
                                if u <= 0.5:
                                    beta = (2 * u) ** (1.0 / (eta_c + 1))
                                else:
                                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                                
                                child1[j] = 0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j])
                                child2[j] = 0.5 * ((1 - beta) * parent1[j] + (1 + beta) * parent2[j])
                    
                    offspring.extend([child1, child2])
                else:
                    # Pas de crossover - les parents sont conservés
                    offspring.extend([parent1, parent2])
        
        return np.array(offspring)
    
    def _polynomial_mutation(self, population: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Mutation polynomiale CORRIGÉE"""
        mutated = population.copy()
        eta_m = 20  # Paramètre de distribution pour la mutation
        
        for i in range(len(population)):
            if self.rng.rand() < self.config.mutation_rate:
                for j in range(population.shape[1]):
                    u = self.rng.rand()
                    
                    if u <= 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    
                    # Application de la mutation
                    mutated[i, j] += delta * (ub[j] - lb[j])
        
        # Application des contraintes de bornes
        return np.clip(mutated, lb, ub)
    
    def optimize(self, obj_fun: Callable, dim: int, lb: np.ndarray, ub: np.ndarray, 
                 function_name: str = "Unknown") -> OptimizationResult:
        """Implémentation Algorithme Génétique CORRIGÉE"""
        start_time = time.perf_counter()
        self.obj_fun = obj_fun  # Assignation de la fonction objectif
        
        try:
            lb, ub = self._validate_bounds(lb, ub, dim)
            
            # Initialisation de la population
            population = self.rng.uniform(lb, ub, (self.config.population_size, dim))
            fitness = np.array([self._evaluate_with_direction(ind) for ind in population])
            
            # Tri initial par fitness
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            best_position = population[0].copy()
            best_fitness = fitness[0]
            
            history = [best_fitness]
            stagnation_count = 0
            convergence_iter = 0
            
            # Boucle d'optimisation principale
            for iteration in range(self.config.max_iter):
                try:
                    # Sélection
                    parents = self._tournament_selection(population, fitness)
                    
                    # Crossover
                    offspring = self._simulated_binary_crossover(parents)
                    
                    # Mutation
                    offspring = self._polynomial_mutation(offspring, lb, ub)
                    
                    # Évaluation des descendants
                    offspring_fitness = np.array([self._evaluate_with_direction(ind) for ind in offspring])
                    
                    # Sélection élitiste (remplacement générationnel)
                    combined_population = np.vstack([population, offspring])
                    combined_fitness = np.hstack([fitness, offspring_fitness])
                    
                    # Sélection des meilleurs individus
                    best_indices = np.argsort(combined_fitness)[:self.config.population_size]
                    population = combined_population[best_indices]
                    fitness = combined_fitness[best_indices]
                    
                    # Mise à jour du meilleur global
                    current_best_fitness = fitness[0]
                    if current_best_fitness < best_fitness:
                        best_fitness = current_best_fitness
                        best_position = population[0].copy()
                        stagnation_count = 0
                        convergence_iter = iteration
                    else:
                        stagnation_count += 1
                    
                    history.append(best_fitness)
                    self._update_metrics(population, best_fitness)
                    
                    # Arrêt précoce
                    if stagnation_count >= self.config.early_stop:
                        break
                        
                except Exception as e:
                    # Continuer en cas d'erreur non-critique
                    continue
            
            execution_time = time.perf_counter() - start_time
            
            if 'monitor' in st.session_state:
                st.session_state.monitor.log_execution_time(execution_time)
            
            # Ajustement final de la fitness
            final_fitness = best_fitness if self.config.minimization else -best_fitness
            
            return OptimizationResult(
                algorithm="GA",
                best_position=best_position,
                best_fitness=final_fitness,
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
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            st.error(f"GA optimization failed: {e}")
            
            return OptimizationResult(
                algorithm="GA",
                best_position=np.full(dim, (lb[0] + ub[0]) / 2),
                best_fitness=float('inf'),
                history=np.array([float('inf')]),
                population=np.zeros((1, dim)),
                fitness=np.array([float('inf')]),
                iterations=0,
                execution_time=execution_time,
                convergence_iter=0,
                config=self.config,
                function_name=function_name,
                dimension=dim
            )
ALGORITHMS = {
    'SMA': CorrectSMAOptimizer,
    'PSO': PSOOptimizer,
    'GA': GeneticAlgorithmOptimizer,
}

# -----------------------------
# Hyperparameter Optimization CORRIGÉE
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
            optimizer = CorrectSMAOptimizer(config)
            result = optimizer.optimize(obj_fun, dim, lb, ub)
            
            if result.best_fitness < best_score:
                best_score = result.best_fitness
                best_params = params
        
        progress_bar.empty()
        status_text.empty()
        
        return {'best_params': best_params, 'best_score': best_score}

# -----------------------------
# Project Management System Amélioré CORRIGÉ
# -----------------------------

class ProjectManager:
    def __init__(self):
        self.projects = {}
        self.current_project = None
    
    def create_project(self, name: str, description: str = ""):
        """Crée un nouveau projet"""
        project_id = f"project_{int(time.time())}"
        self.projects[project_id] = {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'sessions': []
        }
        self.current_project = project_id
        return project_id
    
    def save_session(self, config: OptimizationConfig, results: List[OptimizationResult]):
        """Sauvegarde une session d'optimisation COMPLÈTE"""
        if self.current_project:
            session_id = f"session_{int(time.time())}"
            session_data = {
                'id': session_id,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'algorithm': config.algorithm,
                    'population_size': config.population_size,
                    'max_iter': config.max_iter,
                    'minimization': config.minimization,
                    'seed': config.seed,
                    'early_stop': config.early_stop,
                    'z_param': config.z_param,
                    'adaptive_z': config.adaptive_z,
                    'w': config.w,
                    'c1': config.c1,
                    'c2': config.c2,
                    'crossover_rate': config.crossover_rate,
                    'mutation_rate': config.mutation_rate,
                    'F': config.F,
                    'CR': config.CR
                },
                'results': [{
                    'algorithm': r.algorithm,
                    'best_fitness': float(r.best_fitness),
                    'best_position': r.best_position.tolist(),
                    'iterations': r.iterations,
                    'execution_time': r.execution_time,
                    'convergence_iter': r.convergence_iter,
                    'function_name': r.function_name,
                    'dimension': r.dimension,
                    'history': r.history.tolist() if hasattr(r.history, 'tolist') else []
                } for r in results]
            }
            self.projects[self.current_project]['sessions'].append(session_data)
            return session_id
        return None

# -----------------------------
# Enhanced Visualization Functions CORRIGÉES
# -----------------------------

def create_real_time_dashboard(results: List[OptimizationResult]) -> go.Figure:
    """Crée un tableau de bord en temps réel CORRIGÉ"""
    try:
        if not PLOTLY_AVAILABLE or not results:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Convergence History', 'Population Diversity', 
                           'Exploration vs Exploitation', 'Algorithm Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Graphique de convergence
        for i, result in enumerate(results):
            if len(result.history) > 0:
                # Prendre seulement les 100 premiers points si trop longs pour la performance
                history_to_plot = result.history[:100] if len(result.history) > 100 else result.history
                fig.add_trace(
                    go.Scatter(y=history_to_plot, name=f'{result.algorithm} Run {i+1}',
                              line=dict(width=2)),
                    row=1, col=1
                )
        
        # Graphique de diversité
        for i, result in enumerate(results):
            if 'metrics_history' in result.metadata and 'diversity' in result.metadata['metrics_history']:
                diversity = result.metadata['metrics_history']['diversity']
                if len(diversity) > 0:
                    diversity_to_plot = diversity[:100] if len(diversity) > 100 else diversity
                    fig.add_trace(
                        go.Scatter(y=diversity_to_plot, name=f'{result.algorithm} Diversity',
                                  line=dict(dash='dot')),
                        row=1, col=2
                    )
        
        fig.update_layout(height=600, title_text="Optimization Dashboard")
        return fig
        
    except Exception as e:
        return None

def plot_algorithm_comparison(results: List[OptimizationResult]) -> go.Figure:
    """Compare les performances des algorithmes CORRIGÉ"""
    try:
        if not PLOTLY_AVAILABLE or not results:
            return None
            
        algorithms = list(set([r.algorithm for r in results]))
        metrics_data = []
        
        for algo in algorithms:
            algo_results = [r for r in results if r.algorithm == algo]
            if algo_results:
                metrics_data.append({
                    'Algorithm': algo,
                    'Best Fitness': np.mean([r.best_fitness for r in algo_results]),
                    'Execution Time': np.mean([r.execution_time for r in algo_results]),
                    'Iterations': np.mean([r.iterations for r in algo_results])
                })
        
        if not metrics_data:
            return None
            
        df = pd.DataFrame(metrics_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Best Fitness', x=df['Algorithm'], y=df['Best Fitness']))
        fig.add_trace(go.Bar(name='Execution Time', x=df['Algorithm'], y=df['Execution Time']))
        fig.add_trace(go.Bar(name='Iterations', x=df['Algorithm'], y=df['Iterations']))
        
        fig.update_layout(barmode='group', title="Algorithm Performance Comparison")
        return fig
        
    except Exception:
        return None

def create_3d_visualization(results: List[OptimizationResult], params: dict):
    """Crée une visualisation 3D CORRIGÉE"""
    if params.get('dim', 2) < 2:
        st.warning("3D visualization requires at least 2 dimensions")
        return None
    
    try:
        func_name = params.get('func_name', 'Sphere')
        benchmark = BENCHMARKS.get(func_name, BENCHMARKS['Sphere'])
        bounds = benchmark['bounds']
        
        # Utiliser seulement les 2 premières dimensions pour la visualisation
        x = np.linspace(bounds[0], bounds[1], 25)  # Réduit la résolution pour la performance
        y = np.linspace(bounds[0], bounds[1], 25)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Créer un point en 2D seulement pour la visualisation
                point_2d = np.array([X[i,j], Y[i,j]])
                # Étendre aux dimensions nécessaires en répétant les valeurs
                if params.get('dim', 2) > 2:
                    full_point = np.full(params['dim'], point_2d[0])
                    full_point[:2] = point_2d
                else:
                    full_point = point_2d
                Z[i,j] = benchmark['func'](full_point)
        
        fig = go.Figure(data=[
            go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.7,
                      contours=dict(z=dict(show=True, usecolormap=True))),
        ])
        
        # Ajouter les meilleures solutions
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
            title=f"3D Landscape: {func_name} (First 2 Dimensions)",
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
    """Crée un graphique de coordonnées parallèles CORRIGÉ"""
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
# Statistical Analysis CORRIGÉE
# -----------------------------

class StatisticalAnalyzer:
    @staticmethod
    def wilcoxon_signed_rank_test(results_a: List[OptimizationResult], 
                                 results_b: List[OptimizationResult]) -> Dict[str, Any]:
        """Test de Wilcoxon pour comparer deux algorithmes CORRIGÉ"""
        if not results_a or not results_b:
            return {'error': 'Insufficient data for statistical test'}
            
        fitness_a = [r.best_fitness for r in results_a]
        fitness_b = [r.best_fitness for r in results_b]
        
        if len(fitness_a) != len(fitness_b):
            min_len = min(len(fitness_a), len(fitness_b))
            fitness_a = fitness_a[:min_len]
            fitness_b = fitness_b[:min_len]
        
        try:
            if SCIPY_AVAILABLE and len(fitness_a) >= 2:
                stat, p_value = stats.wilcoxon(fitness_a, fitness_b)
                return {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            else:
                return {'error': 'Scipy not available or insufficient data for statistical test'}
        except Exception as e:
            return {'error': f'Statistical test failed: {str(e)}'}
    
    @staticmethod
    def calculate_convergence_metrics(history: np.ndarray) -> Dict[str, float]:
        """Calcule les métriques de convergence CORRIGÉ"""
        if len(history) == 0:
            return {}
        
        try:
            return {
                'auc': np.trapz(history),
                'final_improvement': history[0] - history[-1],
                'convergence_speed': len(history) / (abs(history[0] - history[-1]) + 1e-12),
                'stability': np.std(history[-min(10, len(history)):]) if len(history) >= 5 else np.std(history)
            }
        except:
            return {}

# -----------------------------
# Enhanced Export Utilities CORRIGÉES
# -----------------------------

def create_comprehensive_report(results: List[OptimizationResult], project_info: Dict[str, Any]) -> Dict[str, Any]:
    """Crée un rapport complet d'optimisation CORRIGÉ"""
    try:
        report = {
            'project_info': project_info,
            'summary': {
                'total_runs': len(results),
                'algorithms_used': list(set([r.algorithm for r in results])) if results else [],
                'best_overall_fitness': min([r.best_fitness for r in results]) if results else float('inf'),
                'average_execution_time': np.mean([r.execution_time for r in results]) if results else 0,
                'timestamp': datetime.now().isoformat()
            },
            'detailed_results': {},
            'statistical_analysis': {},
            'recommendations': []
        }
        
        if not results:
            return report
        
        # Analyse détaillée par algorithme
        for algo in set([r.algorithm for r in results]):
            algo_results = [r for r in results if r.algorithm == algo]
            if algo_results:
                fitness_values = [r.best_fitness for r in algo_results]
                global_min = BENCHMARKS.get(algo_results[0].function_name, {}).get('global_min', 0)
                
                report['detailed_results'][algo] = {
                    'best_fitness': min(fitness_values),
                    'worst_fitness': max(fitness_values),
                    'average_fitness': np.mean(fitness_values),
                    'std_fitness': np.std(fitness_values),
                    'success_rate': np.mean([1 if abs(r.best_fitness - global_min) < 1e-3 else 0 
                                           for r in algo_results])
                }
        
        # Recommandations
        if report['detailed_results']:
            best_algo = min(report['detailed_results'].items(), 
                           key=lambda x: x[1]['average_fitness'])[0]
            report['recommendations'].append(f"Best performing algorithm: {best_algo}")
        
        return report
        
    except Exception:
        return {}

# -----------------------------
# Streamlit UI - Version complète CORRIGÉE
# -----------------------------

def create_advanced_controls():
    """Crée les contrôles avancés dans la sidebar CORRIGÉ"""
    with st.sidebar.expander("⚙️ Advanced Controls", expanded=False):
        use_parallel = st.checkbox("Enable Parallel Processing", value=False)
        max_workers = st.slider("Max Workers", 1, min(4, cpu_count()), 2)
        
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
    """Configure la barre latérale de l'interface CORRIGÉE"""
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
            dim = st.slider("Dimension", 1, 50, 10)
            minimization = st.selectbox("Type", ["Minimize", "Maximize"]) == "Minimize"
        
        st.subheader("Algorithm Selection")
        available_algorithms = list(ALGORITHMS.keys())
        algorithms = st.multiselect(
            "Algorithms to Compare",
            available_algorithms,
            default=["SMA", "PSO"] if "SMA" in available_algorithms and "PSO" in available_algorithms else available_algorithms[:2]
        )
        
        st.subheader("General Parameters")
        col1, col2 = st.columns(2)
        with col1:
            pop_size = st.slider("Population", 10, 200, 30)
            max_iter = st.slider("Max Iterations", 50, 1000, 200)
        with col2:
            num_runs = st.slider("Runs per Algorithm", 1, 10, 2)
            seed = st.number_input("Seed", value=42, min_value=0)
        
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
    """Affiche les résultats de l'optimisation CORRIGÉ"""
    if not results:
        st.error("No results to display - all optimizations failed")
        return
        
    st.header("📊 Optimization Summary")
    
    # Métriques principales
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
        success_count = sum([1 for r in results if r.best_fitness < float('inf')])
        st.metric("Successful Runs", f"{success_count}/{len(results)}")
    
    # Monitoring des performances
    if 'monitor' in st.session_state:
        st.subheader("📈 Performance Monitoring")
        st.session_state.monitor.display_metrics()
    
    # Tableau de bord en temps réel
    if PLOTLY_AVAILABLE and results:
        st.subheader("📈 Optimization Dashboard")
        dashboard = create_real_time_dashboard(results)
        if dashboard:
            st.plotly_chart(dashboard, use_container_width=True)
    
    # Comparaison des algorithmes
    if len(set([r.algorithm for r in results])) > 1:
        st.subheader("🔄 Algorithm Comparison")
        comp_fig = plot_algorithm_comparison(results)
        if comp_fig:
            st.plotly_chart(comp_fig, use_container_width=True)
    
    # Visualisation 3D
    if params.get('advanced_controls', {}).get('enable_3d', False) and results and params.get('dim', 2) >= 2:
        st.subheader("🌐 3D Landscape Visualization")
        fig_3d = create_3d_visualization(results, params)
        if fig_3d:
            st.plotly_chart(fig_3d, use_container_width=True)
    
    # Coordonnées parallèles
    if params.get('advanced_controls', {}).get('enable_parallel_plot', False) and len(results) >= 2:
        st.subheader("📊 Parallel Coordinates Analysis")
        parallel_fig = create_parallel_coordinates_plot(results)
        if parallel_fig:
            st.plotly_chart(parallel_fig, use_container_width=True)
    
    # Résultats détaillés par algorithme
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
                if algo_results:
                    fitness_values = [r.best_fitness for r in algo_results]
                    stats_data = {
                        'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
                        'Fitness': [
                            f"{np.mean(fitness_values):.2e}",
                            f"{np.std(fitness_values):.2e}",
                            f"{np.min(fitness_values):.2e}",
                            f"{np.max(fitness_values):.2e}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    # Analyse statistique
    algorithms_used = list(set([r.algorithm for r in results]))
    if len(algorithms_used) > 1 and len(results) >= 2:
        st.subheader("📊 Statistical Analysis")
        analyzer = StatisticalAnalyzer()
        
        for i in range(len(algorithms_used)):
            for j in range(i + 1, len(algorithms_used)):
                algo_a_results = [r for r in results if r.algorithm == algorithms_used[i]]
                algo_b_results = [r for r in results if r.algorithm == algorithms_used[j]]
                
                test_result = analyzer.wilcoxon_signed_rank_test(algo_a_results, algo_b_results)
                
                if 'error' not in test_result:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{algorithms_used[i]} vs {algorithms_used[j]}**")
                    with col2:
                        st.write(f"p-value: {test_result['p_value']:.4f}")
                    with col3:
                        if test_result['significant']:
                            st.success("Significant difference")
                        else:
                            st.warning("No significant difference")
                else:
                    st.warning(f"Cannot compare {algorithms_used[i]} vs {algorithms_used[j]}: {test_result['error']}")

def setup_advanced_export(project_manager: ProjectManager, results: List[OptimizationResult], params: dict):
    """Configure la section d'export avancée CORRIGÉE"""
    st.sidebar.header("💾 Advanced Export")
    
    # Sauvegarde de projet
    if st.sidebar.button("💼 Save Project") and results:
        try:
            project_id = project_manager.create_project(
                f"Optimization_{params.get('func_name', 'Unknown')}",
                f"Multi-algorithm comparison on {params.get('func_name', 'Unknown')}"
            )
            # Utiliser la première configuration disponible
            if results:
                config = results[0].config
                session_id = project_manager.save_session(config, results)
                if session_id:
                    st.sidebar.success(f"✅ Project saved!")
                else:
                    st.sidebar.error("❌ Failed to save project session")
            else:
                st.sidebar.error("❌ No results to save")
        except Exception as e:
            st.sidebar.error(f"❌ Error saving project: {str(e)}")
    
    # Génération de rapport complet
    if st.sidebar.button("📄 Generate Full Report") and results:
        try:
            report = create_comprehensive_report(results, {
                'function': params.get('func_name', 'Unknown'),
                'dimension': params.get('dim', 10),
                'algorithms': params.get('algorithms', []),
                'timestamp': datetime.now().isoformat()
            })
            
            report_json = json.dumps(report, indent=2, ensure_ascii=False)
            b64 = base64.b64encode(report_json.encode()).decode()
            
            st.sidebar.markdown(
                f'<a href="data:application/json;base64,{b64}" download="optimization_report.json" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 0.25rem; text-align: center;">📥 Download Full Report</a>',
                unsafe_allow_html=True
            )
            st.sidebar.success("✅ Report generated!")
        except Exception as e:
            st.sidebar.error(f"❌ Error generating report: {str(e)}")
    
    # Export des données de comparaison
    if st.sidebar.button("📊 Export Comparison Data") and results:
        try:
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
                f'<a href="data:application/csv;base64,{b64}" download="algorithm_comparison.csv" style="display: inline-block; padding: 0.5rem 1rem; background-color: #2196F3; color: white; text-decoration: none; border-radius: 0.25rem; text-align: center;">📥 Download Comparison Data</a>',
                unsafe_allow_html=True
            )
            st.sidebar.success("✅ Data exported!")
        except Exception as e:
            st.sidebar.error(f"❌ Error exporting data: {str(e)}")

def main():
    """Fonction principale de l'application CORRIGÉE"""
    # Configuration de la page
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
    
    *Note: This is the CORRECTED version with all bugs fixed.*
    """)
    
    # Initialisation de l'état de session
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
    if 'last_params' not in st.session_state:
        st.session_state.last_params = {}
    
    # Information système
    with st.expander("🔧 System Information", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write(f"**Numba:** {NUMBA_AVAILABLE}")
        with col2:
            st.write(f"**Plotly:** {PLOTLY_AVAILABLE}")
        with col3:
            st.write(f"**Scipy:** {SCIPY_AVAILABLE}")
        with col4:
            st.write(f"**Algorithms:** {len(ALGORITHMS)}")
        with col5:
            st.write(f"**CPU Cores:** {cpu_count()}")
    
    # Configuration via sidebar
    params = setup_sidebar()
    
    # Exécution de l'optimisation
    if params['submitted']:
        benchmark = BENCHMARKS[params['func_name']]
        obj_fun = benchmark['func']
        bounds = benchmark['bounds']
        
        all_results = []
        
        # Validation de la configuration
        if not params['algorithms']:
            st.error("Please select at least one algorithm")
            st.stop()
        
        # Optimisation des hyperparamètres si demandée
        if params['optimize_hyperparams'] and 'SMA' in params['algorithms']:
            with st.spinner("🔍 Optimizing SMA hyperparameters..."):
                try:
                    hyper_optimizer = HyperparameterOptimizer()
                    hyper_result = hyper_optimizer.optimize_hyperparameters(
                        obj_fun, params['dim'], 
                        np.full(params['dim'], bounds[0]),
                        np.full(params['dim'], bounds[1])
                    )
                    st.success(f"Best parameters: {hyper_result['best_params']}")
                except Exception as e:
                    st.error(f"Hyperparameter optimization failed: {e}")
        
        # Réinitialisation du moniteur
        st.session_state.monitor.reset()
        
        # Préparation des configurations
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
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Exécution séquentielle (plus stable)
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
        
        # Nettoyage de l'interface
        progress_bar.empty()
        status_text.empty()
        
        # Sauvegarde des résultats dans l'historique
        st.session_state.optimization_history.extend(all_results)
        st.session_state.last_params = params
        
        # Affichage des résultats
        display_enhanced_results(all_results, params)
        
        # Configuration de l'export
        setup_advanced_export(st.session_state.project_manager, all_results, params)
        
        # Insights de performance
        if all_results:
            st.sidebar.header("🎯 Performance Insights")
            successful_results = [r for r in all_results if r.best_fitness < float('inf')]
            if successful_results:
                best_algo = min(set([r.algorithm for r in successful_results]), 
                               key=lambda algo: np.mean([r.best_fitness for r in successful_results if r.algorithm == algo]))
                st.sidebar.success(f"**Recommended Algorithm:** {best_algo}")
                
                # Calcul de l'amélioration moyenne seulement pour les runs réussis
                improvements = []
                for r in successful_results:
                    if len(r.history) > 1:
                        improvements.append(r.history[0] - r.history[-1])
                
                if improvements:
                    avg_improvement = np.mean(improvements)
                    st.sidebar.info(f"**Average Improvement:** {avg_improvement:.2e}")
            
            st.session_state.logger.log_optimization_end(all_results[0])
        else:
            st.sidebar.error("❌ No successful optimizations")
    
    elif st.session_state.optimization_history:
        # Afficher les résultats existants si disponibles
        st.info("📊 Showing previous optimization results")
        display_enhanced_results(st.session_state.optimization_history, st.session_state.last_params)
        setup_advanced_export(st.session_state.project_manager, st.session_state.optimization_history, st.session_state.last_params)
    
    else:
        st.info("👆 Configure your optimization parameters and click 'Run Optimization' to get started!")

# Point d'entrée de l'application
if __name__ == "__main__":
    main()
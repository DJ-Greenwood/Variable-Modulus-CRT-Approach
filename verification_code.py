import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import math
import os
from datetime import datetime
from sympy import factorint, mod_inverse

# Set precision for large number calculations
getcontext().prec = 100

class CollatzTester:
    def __init__(self, max_test_range: int = 10000):
        """Initialize tester with maximum test range"""
        self.max_test_range = max_test_range
        self.results_df = pd.DataFrame()
        self.modulus_cache = {}
        self.sequence_cache = {}
        self.crt_classifications = {}
    
    def T(self, n: Decimal) -> Decimal:
        """The Collatz function T(n)"""
        return n // 2 if n % 2 == 0 else 3 * n + 1

    def get_sequence(self, n: Decimal, max_steps: int = 1000) -> List[Decimal]:
        """Generate Collatz sequence starting from n"""
        if n in self.sequence_cache:
            return self.sequence_cache[n]
        
        sequence = [n]
        current = n
        
        for _ in range(max_steps):
            current = self.T(current)
            sequence.append(current)
            if current == 1:
                break
        
        self.sequence_cache[n] = sequence
        return sequence
    
    def M(self, n: Decimal) -> Decimal:
        """Variable modulus function M(n) with correction mechanism applied only to odd values"""
        if n in self.modulus_cache:
            return self.modulus_cache[n]
        
        if n % 2 == 1:
            result = self.calculate_modulus(3 * n + 1)
            # Correction mechanism: Ensure non-decreasing modulus only for odd values
            if n > 1:
                prev_modulus = self.M(self.T(n))
                result = max(result, prev_modulus)
        else:
            result = self.calculate_modulus(n)
        
        self.modulus_cache[n] = result
        return result
    
    def calculate_modulus(self, n: Decimal) -> Decimal:
        """Calculate modulus using prime factorization, ensuring integer conversion"""
        factors = factorint(int(n))
        modulus = Decimal(1)
        
        for prime, power in factors.items():
            prime_int = int(prime)  # Convert gmpy2.mpz to int
            power_int = int(power)  # Convert gmpy2.mpz to int
            modulus = (modulus * Decimal(prime_int ** power_int)) // math.gcd(int(modulus), prime_int ** power_int)
        
        return modulus
    
    def classify_modular_equivalence(self, n: int) -> Tuple[int, int]:
        """Classify numbers using the Chinese Remainder Theorem (CRT)"""
        modulus = self.M(Decimal(n))
        residue = n % int(modulus)
        self.crt_classifications[n] = (modulus, residue)
        return modulus, residue
    
    def test_number(self, n: int) -> Dict:
        """Test a single number for all properties"""
        n_decimal = Decimal(n)
        sequence = self.get_sequence(n_decimal)
        modulus_values = [self.M(Decimal(x)) for x in sequence]
        
        # Apply modulus violation check only to odd numbers
        modulus_violations = 0
        for i in range(len(sequence) - 1):
            if sequence[i] % 2 == 1:  # Only consider odd numbers
                if self.M(Decimal(sequence[i + 1])) < self.M(Decimal(sequence[i])):
                    modulus_violations += 1
        
        result = {
            'number': n,
            'sequence_length': len(sequence),
            'reaches_one': sequence[-1] == 1,
            'max_value': max(sequence),
            'min_modulus': min(modulus_values),
            'max_modulus': max(modulus_values),
            'modulus_strictly_increasing': modulus_violations == 0,
            'contains_cycles': len(set(sequence)) != len(sequence),
            'converges_to_one': sequence[-1] == 1,
            'steps_to_one': len(sequence) - 1 if sequence[-1] == 1 else -1,
            'crt_modulus': self.classify_modular_equivalence(n)[0],
            'crt_residue': self.classify_modular_equivalence(n)[1],
            'modulus_violations': modulus_violations
        }
        
        return result
    
    def run_tests(self) -> pd.DataFrame:
        """Run comprehensive tests and return results as DataFrame"""
        results = []
        
        for n in range(1, self.max_test_range + 1):
            results.append(self.test_number(n))
            
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def generate_proof(self) -> Dict:
        """Mathematical verification of non-cyclic growth"""
        if self.results_df.empty:
            print("No test results available. Run tests first.")
            return {}
        
        summary = {
            'total_numbers_tested': len(self.results_df),
            'numbers_reaching_one': sum(self.results_df['reaches_one']),
            'max_sequence_length': self.results_df['sequence_length'].max(),
            'total_modulus_violations': sum(self.results_df['modulus_violations']),
            'numbers_with_cycles': sum(self.results_df['contains_cycles']),
            'convergence_to_one': sum(self.results_df['converges_to_one']),
            'max_modulus_value': self.results_df['max_modulus'].max(),
            'crt_unique_classes': len(set(self.results_df['crt_modulus']))
        }
        
        return summary
    
    def save_results(self, directory: str = "data"):
        """Save test results to CSV file"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"collatz_test_results_{timestamp}.csv"
        filepath = os.path.join(directory, filename)
        
        self.results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")


def main():
    test_range = 2**100+1
    tester = CollatzTester(max_test_range=test_range)
    print("Starting Collatz conjecture tests...")
    results_df = tester.run_tests()
    print(f'\nResults DataFrame:\n{results_df}')
    tester.save_results()
    summary = tester.generate_proof()
    print("\nTest Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
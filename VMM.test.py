import math
from decimal import Decimal, getcontext
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import pandas as pd

# Set much higher precision for large number calculations
getcontext().prec = 10_000_000  # Increased precision for very large numbers

class CollatzValidator:
    def __init__(self, max_test_range: int = 10000, verbose: bool = True):
        """Initialize validator with maximum test range"""
        self.max_test_range = Decimal(max_test_range)
        self.modulus_cache = {}  # Cache for M(n) values
        self.sequence_cache = {}  # Cache for Collatz sequences
        self.verbose = verbose

    def log(self, message: str):
        """Print debug message if verbose mode is on"""
        if self.verbose:
            print(message)

    def T(self, n: Decimal) -> Decimal:
        """The Collatz function T(n) with proper Decimal handling"""
        is_even = (n % 2) == 0
        if is_even:
            result = n / 2
        else:
            result = 3 * n + 1
        self.log(f"T({n}) = {result} {'(even step)' if is_even else '(odd step)'}")
        return result

    def get_sequence(self, n: Decimal, max_steps: int = 1000) -> List[Decimal]:
        """Generate Collatz sequence starting from n"""
        self.log(f"\nGenerating sequence for n = {n}")
        
        if n in self.sequence_cache:
            self.log("Using cached sequence")
            return self.sequence_cache[n]

        sequence = [n]
        current = n
        step = 0
        
        for _ in range(max_steps):
            step += 1
            current = self.T(current)
            sequence.append(current)
            if current == 1:
                self.log(f"Sequence reached 1 in {step} steps")
                break
                
        self.sequence_cache[n] = sequence
        return sequence

    def gcd(self, a: Decimal, b: Decimal) -> Decimal:
        """Calculate GCD of two Decimal numbers"""
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a

    def lcm(self, a: Decimal, b: Decimal) -> Decimal:
        """Calculate LCM of two Decimal numbers"""
        gcd = self.gcd(a, b)
        if gcd == 0:
            return Decimal(0)
        return abs(a * b) / gcd

    def M(self, n: Decimal) -> Decimal:
        """Variable modulus function M(n) defined in the paper"""
        self.log(f"\nCalculating M({n})")
        
        if n in self.modulus_cache:
            self.log(f"Using cached M({n}) = {self.modulus_cache[n]}")
            return self.modulus_cache[n]

        # For odd n, calculate based on prime factors of 3n + 1
        if n % 2 == 1:
            self.log(f"n is odd, calculating M(n) based on 3n + 1 = {3 * n + 1}")
            result = self.calculate_modulus(3 * n + 1)
        else:
            self.log(f"n is even, calculating M(n) directly")
            result = self.calculate_modulus(n)

        self.modulus_cache[n] = result
        self.log(f"M({n}) = {result}")
        return result

    def calculate_modulus(self, n: Decimal) -> Decimal:
        """Calculate modulus based on prime factorization with Decimal handling"""
        self.log(f"\nCalculating modulus for {n}")
        factors = self.prime_factorize(n)
        self.log(f"Prime factorization: {dict(factors)}")
        
        modulus = Decimal(1)
        self.log("Calculating LCM of prime powers:")
        
        # Calculate LCM of prime powers
        for prime, power in factors.items():
            prime_power = Decimal(prime) ** Decimal(power)
            modulus = self.lcm(modulus, prime_power)
            self.log(f"  LCM after including {prime}^{power}: {modulus}")
            
        return modulus

    def prime_factorize(self, n: Decimal) -> Dict[int, int]:
        """Get prime factorization of n with large number handling"""
        self.log(f"\nCalculating prime factorization of {n}")
        factors = defaultdict(int)
        num = n
        
        # Handle 2 separately
        while (num % 2) == 0:
            factors[2] += 1
            num = num / 2
            self.log(f"  Found factor 2, remaining: {num}")

        # Check odd factors
        i = Decimal(3)
        while i * i <= num:
            while (num % i) == 0:
                factors[int(i)] += 1
                num = num / i
                self.log(f"  Found factor {i}, remaining: {num}")
            i += 2

        if num > 2:
            factors[int(num)] += 1
            self.log(f"  Final factor: {num}")

        return dict(factors)

    def verify_modulus_growth(self, n: Decimal) -> bool:
        """Verify Lemma 2.3: Modulus Growth Property"""
        self.log(f"\nVerifying modulus growth for n = {n}")
        
        if n % 2 == 1:  # Only check odd numbers
            M_n = self.M(n)
            T_n = self.T(n)
            M_Tn = self.M(T_n)
            
            self.log(f"Comparing M(T({n})) = {M_Tn} with M({n}) = {M_n}")
            result = M_Tn >= M_n
            self.log(f"Modulus growth {'satisfied' if result else 'violated'}")
            return result
            
        self.log("Skipping even number")
        return True

    def verify_no_cycles(self, n: Decimal, max_steps: int = 1000) -> bool:
        """Verify Theorem 3.2: No Infinite Cycles"""
        self.log(f"\nVerifying no cycles for n = {n}")
        sequence = self.get_sequence(n, max_steps)
        seen = set()
        
        for num in sequence:
            if num in seen and num != 1:
                self.log(f"Cycle detected! Number {num} appeared twice")
                return False
            seen.add(num)
            
        self.log("No cycles detected")
        return True

    def verify_convergence(self, n: Decimal, max_steps: int = 1000) -> bool:
        """Verify convergence to 1"""
        self.log(f"\nVerifying convergence for n = {n}")
        sequence = self.get_sequence(n, max_steps)
        result = sequence[-1] == 1
        self.log(f"Convergence to 1: {'success' if result else 'failure'}")
        return result

    def run_validation(self, start_range: int, test_range: int = None) -> pd.DataFrame:
        """Run comprehensive validation of the paper's claims"""
        start = Decimal(start_range)
        end = Decimal(test_range if test_range is not None else self.max_test_range)
            
        self.log(f"\nStarting validation for numbers {start} to {end}")
        
        results = []

        current = start
        while current <= end:
            self.log(f"\n{'='*50}")
            self.log(f"Testing number {current}")
            self.log('='*50)
            
            result = {
                'number': int(current),
                'modulus_growth': self.verify_modulus_growth(current),
                'no_cycles': self.verify_no_cycles(current),
                'convergence': self.verify_convergence(current)
            }
            results.append(result)

            current += 1

        df = pd.DataFrame(results)
        return df
    
    def save_report(self, df: pd.DataFrame, filename: str):
        """Save validation results to a CSV file"""
        df.to_csv(filename, index=False)
        self.log(f"Validation results saved to {filename}")

def main():
    # Create validator instance with verbose output
    validator = CollatzValidator(max_test_range=1000000000000000000, verbose=True)
    
    # Test with a very large number (2^105 + 1 to 2^105 + 10)
    start = 1
    end = start + 2**10   #  Decimal(2) ** Decimal(100) + 1
    
    print(f"Starting detailed validation for range {start} to {end}...")
    df = validator.run_validation(start_range=start, test_range=end)
    
    # Print summary results
    print("\nValidation Results:")
    print(df)

    # Save results to a CSV file
    validator.save_report(df, "collatz_validation_results.csv")

if __name__ == "__main__":
    main()

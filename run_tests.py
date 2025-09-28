#!/usr/bin/env python3
# Copyright © 2025 Manus AI

"""
Test runner for MLX Speculative Decoding

This script provides a comprehensive test runner with options for different
test categories, coverage reporting, and performance benchmarking.
"""

import argparse
import sys
import os
import unittest
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.fixtures.test_utils import test_metrics


def discover_tests(test_dir: str, pattern: str = "test_*.py") -> unittest.TestSuite:
    """Discover tests in a directory."""
    loader = unittest.TestLoader()
    start_dir = os.path.join("tests", test_dir)
    
    if not os.path.exists(start_dir):
        print(f"Warning: Test directory '{start_dir}' not found")
        return unittest.TestSuite()
    
    suite = loader.discover(start_dir, pattern=pattern)
    return suite


def run_test_suite(suite: unittest.TestSuite, verbosity: int = 2) -> unittest.TestResult:
    """Run a test suite and return results."""
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        stream=sys.stdout,
        buffer=True,
    )
    
    result = runner.run(suite)
    return result


def run_unit_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run unit tests."""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    suite = discover_tests("unit")
    return run_test_suite(suite, verbosity)


def run_integration_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run integration tests."""
    print("=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)
    
    suite = discover_tests("integration")
    return run_test_suite(suite, verbosity)


def run_performance_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run performance tests."""
    print("=" * 60)
    print("RUNNING PERFORMANCE TESTS")
    print("=" * 60)
    
    suite = discover_tests("performance")
    return run_test_suite(suite, verbosity)


def run_all_tests(verbosity: int = 2) -> dict:
    """Run all test categories."""
    results = {}
    
    # Run unit tests
    results["unit"] = run_unit_tests(verbosity)
    
    # Run integration tests
    results["integration"] = run_integration_tests(verbosity)
    
    # Run performance tests
    results["performance"] = run_performance_tests(verbosity)
    
    return results


def print_summary(results: dict):
    """Print test results summary."""
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    for category, result in results.items():
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        total_skipped += skipped
        
        status = "PASS" if (failures == 0 and errors == 0) else "FAIL"
        
        print(f"{category.upper():12} | {tests_run:3d} tests | "
              f"{failures:2d} failures | {errors:2d} errors | "
              f"{skipped:2d} skipped | {status}")
    
    print("-" * 60)
    overall_status = "PASS" if (total_failures == 0 and total_errors == 0) else "FAIL"
    print(f"{'TOTAL':12} | {total_tests:3d} tests | "
          f"{total_failures:2d} failures | {total_errors:2d} errors | "
          f"{total_skipped:2d} skipped | {overall_status}")
    
    # Print performance metrics if available
    test_metrics.print_summary()
    
    return overall_status == "PASS"


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import mlx.core
    except ImportError:
        missing_deps.append("mlx")
    
    try:
        import fastapi
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing_deps.append("uvicorn")
    
    if missing_deps:
        print("Error: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies and try again.")
        return False
    
    return True


def setup_test_environment():
    """Set up test environment variables."""
    # Set test-specific environment variables
    os.environ.setdefault("TEST_MODEL_PATH", "microsoft/Phi-3-mini-4k-instruct")
    os.environ.setdefault("MAX_TEST_TOKENS", "20")
    os.environ.setdefault("TEST_TIMEOUT", "10.0")
    
    # Disable slow tests by default in CI
    if os.environ.get("CI"):
        os.environ.setdefault("SKIP_SLOW_TESTS", "true")
        os.environ.setdefault("SKIP_MODEL_TESTS", "true")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="MLX Speculative Decoding Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run only unit tests
  python run_tests.py --integration      # Run only integration tests
  python run_tests.py --performance      # Run only performance tests
  python run_tests.py --quick            # Run quick tests only
  python run_tests.py --verbose          # Run with high verbosity
        """
    )
    
    # Test category options
    parser.add_argument(
        "--unit", action="store_true",
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration", action="store_true",
        help="Run integration tests only"
    )
    parser.add_argument(
        "--performance", action="store_true",
        help="Run performance tests only"
    )
    parser.add_argument(
        "--all", action="store_true", default=True,
        help="Run all tests (default)"
    )
    
    # Test options
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip slow tests (sets SKIP_SLOW_TESTS=true)"
    )
    parser.add_argument(
        "--no-models", action="store_true",
        help="Skip model-dependent tests (sets SKIP_MODEL_TESTS=true)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Increase verbosity"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Decrease verbosity"
    )
    
    # Coverage options
    parser.add_argument(
        "--coverage", action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--coverage-html", action="store_true",
        help="Generate HTML coverage report"
    )
    
    # Other options
    parser.add_argument(
        "--failfast", "-f", action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--pattern", default="test_*.py",
        help="Test file pattern (default: test_*.py)"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    setup_test_environment()
    
    # Handle quick mode
    if args.quick:
        os.environ["SKIP_SLOW_TESTS"] = "true"
    
    # Handle no-models mode
    if args.no_models:
        os.environ["SKIP_MODEL_TESTS"] = "true"
    
    # Set verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    print("MLX Speculative Decoding Test Runner")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Test pattern: {args.pattern}")
    
    if os.environ.get("SKIP_SLOW_TESTS") == "true":
        print("Note: Skipping slow tests")
    if os.environ.get("SKIP_MODEL_TESTS") == "true":
        print("Note: Skipping model-dependent tests")
    
    print()
    
    # Record start time
    start_time = time.time()
    
    # Determine which tests to run
    run_specific = args.unit or args.integration or args.performance
    
    try:
        if args.coverage or args.coverage_html:
            # Run with coverage
            try:
                import coverage
                cov = coverage.Coverage()
                cov.start()
            except ImportError:
                print("Error: coverage package not installed")
                print("Install with: pip install coverage")
                return 1
        
        # Run tests
        if run_specific:
            results = {}
            if args.unit:
                results["unit"] = run_unit_tests(verbosity)
            if args.integration:
                results["integration"] = run_integration_tests(verbosity)
            if args.performance:
                results["performance"] = run_performance_tests(verbosity)
        else:
            results = run_all_tests(verbosity)
        
        if args.coverage or args.coverage_html:
            cov.stop()
            cov.save()
            
            print("\n" + "=" * 60)
            print("COVERAGE REPORT")
            print("=" * 60)
            cov.report()
            
            if args.coverage_html:
                cov.html_report(directory="htmlcov")
                print("\nHTML coverage report generated in 'htmlcov/' directory")
    
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\nError running tests: {e}")
        return 1
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    success = print_summary(results)
    
    print(f"\nTotal test time: {total_time:.2f} seconds")
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

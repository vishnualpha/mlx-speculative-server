#!/usr/bin/env python3
"""
Stress Test for MLX Speculative Server

This script performs intensive stress testing to measure:
1. Maximum concurrent request capacity
2. Performance degradation under load
3. Memory usage patterns
4. Speculative decoding efficiency at scale
"""

import asyncio
import aiohttp
import time
import json
import psutil
import os
from typing import List, Dict, Any
import argparse
from dataclasses import dataclass
import signal
import sys


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    server_url: str = "http://localhost:8000"
    max_concurrency: int = 100
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    max_tokens: int = 50
    temperature: float = 0.7
    monitor_interval: float = 1.0


class StressTester:
    """Comprehensive stress tester for the MLX Speculative Server."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.running = True
        self.results = []
        self.metrics_history = []
        self.start_time = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def send_request(self, session: aiohttp.ClientSession, request_id: int, prompt: str) -> Dict[str, Any]:
        """Send a single request and measure performance."""
        start_time = time.time()
        
        data = {
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }
        
        try:
            async with session.post(f"{self.config.server_url}/generate", json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()
                    
                    return {
                        "request_id": request_id,
                        "success": True,
                        "start_time": start_time,
                        "end_time": end_time,
                        "elapsed_time": end_time - start_time,
                        "server_elapsed": result.get("performance", {}).get("elapsed_time", 0),
                        "throughput": result.get("performance", {}).get("throughput", 0),
                        "tokens": result.get("usage", {}).get("completion_tokens", 0),
                        "speculative_stats": result.get("speculative_stats", {}),
                        "prompt_length": len(prompt),
                    }
                else:
                    return {
                        "request_id": request_id,
                        "success": False,
                        "error": f"HTTP {response.status}",
                        "elapsed_time": time.time() - start_time,
                    }
                    
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }
    
    async def monitor_system_metrics(self):
        """Monitor system metrics during the test."""
        while self.running:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                # Try to get server stats
                server_stats = {}
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.config.server_url}/stats") as response:
                            if response.status == 200:
                                server_stats = await response.json()
                except:
                    pass
                
                metrics = {
                    "timestamp": time.time() - self.start_time,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_available_gb": memory.available / (1024**3),
                    "server_stats": server_stats,
                }
                
                self.metrics_history.append(metrics)
                
                await asyncio.sleep(self.config.monitor_interval)
                
            except Exception as e:
                print(f"⚠️ Error monitoring metrics: {e}")
                await asyncio.sleep(self.config.monitor_interval)
    
    async def ramp_up_load(self, session: aiohttp.ClientSession, prompts: List[str]):
        """Gradually ramp up the load to target concurrency."""
        print(f"📈 Ramping up load over {self.config.ramp_up_seconds}s...")
        
        ramp_start = time.time()
        request_id = 0
        
        while (time.time() - ramp_start) < self.config.ramp_up_seconds and self.running:
            # Calculate current target concurrency based on ramp-up progress
            progress = (time.time() - ramp_start) / self.config.ramp_up_seconds
            current_concurrency = int(self.config.max_concurrency * progress)
            
            # Launch requests up to current concurrency
            tasks = []
            for _ in range(min(5, current_concurrency)):  # Launch in small batches
                prompt = prompts[request_id % len(prompts)]
                task = asyncio.create_task(self.send_request(session, request_id, prompt))
                tasks.append(task)
                request_id += 1
            
            # Wait for batch to complete
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in batch_results:
                    if isinstance(result, dict):
                        self.results.append(result)
            
            await asyncio.sleep(0.1)  # Small delay between batches
    
    async def sustained_load(self, session: aiohttp.ClientSession, prompts: List[str]):
        """Run sustained load at maximum concurrency."""
        print(f"🔥 Running sustained load at {self.config.max_concurrency} concurrent requests...")
        
        sustained_start = time.time()
        request_id = len(self.results)
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        
        async def bounded_request(prompt, req_id):
            async with semaphore:
                return await self.send_request(session, req_id, prompt)
        
        while (time.time() - sustained_start) < self.config.duration_seconds and self.running:
            # Launch a batch of requests
            tasks = []
            batch_size = min(20, self.config.max_concurrency)  # Launch in batches
            
            for _ in range(batch_size):
                prompt = prompts[request_id % len(prompts)]
                task = asyncio.create_task(bounded_request(prompt, request_id))
                tasks.append(task)
                request_id += 1
            
            # Wait for batch with timeout
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=30.0
                )
                
                for result in batch_results:
                    if isinstance(result, dict):
                        self.results.append(result)
                        
            except asyncio.TimeoutError:
                print("⚠️ Batch timeout, continuing...")
            
            # Brief pause to prevent overwhelming
            await asyncio.sleep(0.1)
    
    async def run_stress_test(self, prompts: List[str]) -> Dict[str, Any]:
        """Run the complete stress test."""
        print("🚀 Starting MLX Speculative Server Stress Test")
        print("=" * 50)
        print(f"Server: {self.config.server_url}")
        print(f"Max Concurrency: {self.config.max_concurrency}")
        print(f"Duration: {self.config.duration_seconds}s")
        print(f"Ramp-up: {self.config.ramp_up_seconds}s")
        print("")
        
        self.start_time = time.time()
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self.monitor_system_metrics())
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            ) as session:
                
                # Phase 1: Ramp up
                await self.ramp_up_load(session, prompts)
                
                # Phase 2: Sustained load
                if self.running:
                    await self.sustained_load(session, prompts)
        
        finally:
            self.running = False
            monitor_task.cancel()
            
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        total_elapsed = time.time() - self.start_time
        
        return self.analyze_results(total_elapsed)
    
    def analyze_results(self, total_elapsed: float) -> Dict[str, Any]:
        """Analyze stress test results."""
        successful_results = [r for r in self.results if r.get("success", False)]
        failed_results = [r for r in self.results if not r.get("success", False)]
        
        if not successful_results:
            return {"error": "No successful requests"}
        
        # Basic metrics
        total_requests = len(self.results)
        success_rate = len(successful_results) / total_requests if total_requests > 0 else 0
        
        # Performance metrics
        throughputs = [r["throughput"] for r in successful_results if r["throughput"] > 0]
        server_times = [r["server_elapsed"] for r in successful_results if r["server_elapsed"] > 0]
        client_times = [r["elapsed_time"] for r in successful_results]
        total_tokens = sum(r["tokens"] for r in successful_results)
        
        # Speculative decoding metrics
        acceptance_rates = []
        speedups = []
        for r in successful_results:
            spec_stats = r.get("speculative_stats", {})
            if spec_stats.get("acceptance_rate"):
                acceptance_rates.append(spec_stats["acceptance_rate"])
            if spec_stats.get("speedup"):
                speedups.append(spec_stats["speedup"])
        
        # Time-based analysis
        time_buckets = {}
        for r in successful_results:
            bucket = int((r["start_time"] - self.start_time) // 10) * 10  # 10-second buckets
            if bucket not in time_buckets:
                time_buckets[bucket] = []
            time_buckets[bucket].append(r)
        
        return {
            "test_summary": {
                "total_elapsed": total_elapsed,
                "total_requests": total_requests,
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": success_rate,
                "requests_per_second": total_requests / total_elapsed,
            },
            "performance": {
                "total_tokens": total_tokens,
                "tokens_per_second": total_tokens / total_elapsed,
                "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
                "min_throughput": min(throughputs) if throughputs else 0,
                "max_throughput": max(throughputs) if throughputs else 0,
                "avg_server_time": sum(server_times) / len(server_times) if server_times else 0,
                "avg_client_time": sum(client_times) / len(client_times) if client_times else 0,
            },
            "speculative_decoding": {
                "avg_acceptance_rate": sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0,
                "avg_speedup": sum(speedups) / len(speedups) if speedups else 0,
                "min_speedup": min(speedups) if speedups else 0,
                "max_speedup": max(speedups) if speedups else 0,
            },
            "system_metrics": self.metrics_history,
            "time_analysis": {
                bucket: {
                    "requests": len(requests),
                    "avg_throughput": sum(r["throughput"] for r in requests) / len(requests),
                    "success_rate": len([r for r in requests if r["success"]]) / len(requests),
                }
                for bucket, requests in time_buckets.items()
            },
            "errors": [r["error"] for r in failed_results if "error" in r],
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted stress test results."""
        print("\n" + "="*60)
        print("📊 STRESS TEST RESULTS")
        print("="*60)
        
        summary = results["test_summary"]
        perf = results["performance"]
        spec = results["speculative_decoding"]
        
        print(f"\n🎯 Test Summary:")
        print(f"  Duration: {summary['total_elapsed']:.1f}s")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  Requests/sec: {summary['requests_per_second']:.1f}")
        
        print(f"\n⚡ Performance:")
        print(f"  Total Tokens: {perf['total_tokens']}")
        print(f"  Total Throughput: {perf['tokens_per_second']:.1f} tok/s")
        print(f"  Avg Per-Request Throughput: {perf['avg_throughput']:.1f} tok/s")
        print(f"  Throughput Range: {perf['min_throughput']:.1f} - {perf['max_throughput']:.1f} tok/s")
        print(f"  Avg Server Time: {perf['avg_server_time']:.2f}s")
        print(f"  Avg Client Time: {perf['avg_client_time']:.2f}s")
        
        print(f"\n🚀 Speculative Decoding:")
        print(f"  Avg Acceptance Rate: {spec['avg_acceptance_rate']:.1%}")
        print(f"  Avg Speedup: {spec['avg_speedup']:.1f}x")
        print(f"  Speedup Range: {spec['min_speedup']:.1f}x - {spec['max_speedup']:.1f}x")
        
        if results.get("errors"):
            print(f"\n❌ Errors ({len(results['errors'])}):")
            error_counts = {}
            for error in results["errors"]:
                error_counts[error] = error_counts.get(error, 0) + 1
            for error, count in error_counts.items():
                print(f"  {error}: {count}")


# Test prompts for stress testing
STRESS_TEST_PROMPTS = [
    "What is AI?",
    "Explain ML.",
    "How does deep learning work?",
    "What is NLP?",
    "Describe computer vision.",
    "What is reinforcement learning?",
    "How do neural networks work?",
    "What is machine learning?",
    "Explain artificial intelligence.",
    "What is data science?",
] * 5  # Repeat for variety


async def main():
    """Main stress test function."""
    parser = argparse.ArgumentParser(description="Stress test MLX Speculative Server")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--ramp-up", type=int, default=10, help="Ramp-up time in seconds")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    
    args = parser.parse_args()
    
    config = StressTestConfig(
        server_url=args.server,
        max_concurrency=args.concurrency,
        duration_seconds=args.duration,
        ramp_up_seconds=args.ramp_up,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    # Check server health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{config.server_url}/health") as response:
                if response.status != 200:
                    print(f"❌ Server not healthy: HTTP {response.status}")
                    return
                print("✅ Server is healthy")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    tester = StressTester(config)
    
    try:
        results = await tester.run_stress_test(STRESS_TEST_PROMPTS)
        tester.print_results(results)
        
        # Save results to file
        timestamp = int(time.time())
        filename = f"stress_test_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

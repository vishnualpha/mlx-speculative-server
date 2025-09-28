#!/usr/bin/env python3
"""
Concurrent Request Testing for MLX Speculative Server

This script tests the server's ability to handle multiple concurrent requests
and measures the performance benefits of speculative decoding under load.
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict, Any
import argparse


class ConcurrentTester:
    """Test concurrent request performance."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.results = []
    
    async def single_request(
        self, 
        session: aiohttp.ClientSession, 
        prompt: str, 
        request_id: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a single generation request."""
        start_time = time.time()
        
        data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }
        
        try:
            async with session.post(f"{self.server_url}/generate", json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()
                    
                    return {
                        "request_id": request_id,
                        "success": True,
                        "elapsed_time": end_time - start_time,
                        "server_elapsed": result.get("performance", {}).get("elapsed_time", 0),
                        "throughput": result.get("performance", {}).get("throughput", 0),
                        "tokens_generated": result.get("usage", {}).get("completion_tokens", 0),
                        "speculative_stats": result.get("speculative_stats", {}),
                        "text_length": len(result.get("text", "")),
                        "response": result,
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
    
    async def concurrent_test(
        self,
        prompts: List[str],
        concurrency: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Run concurrent requests test."""
        print(f"🚀 Starting concurrent test with {concurrency} requests...")
        print(f"📝 Using {len(prompts)} different prompts")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(session, prompt, request_id):
            async with semaphore:
                return await self.single_request(session, prompt, request_id, **kwargs)
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Create tasks for concurrent requests
            tasks = []
            for i in range(concurrency):
                prompt = prompts[i % len(prompts)]  # Cycle through prompts
                task = bounded_request(session, prompt, i)
                tasks.append(task)
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_elapsed = end_time - start_time
        
        # Analyze results
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        if successful_results:
            throughputs = [r["throughput"] for r in successful_results if r["throughput"] > 0]
            server_times = [r["server_elapsed"] for r in successful_results if r["server_elapsed"] > 0]
            client_times = [r["elapsed_time"] for r in successful_results]
            total_tokens = sum(r["tokens_generated"] for r in successful_results)
            
            # Speculative decoding stats
            acceptance_rates = []
            speedups = []
            for r in successful_results:
                spec_stats = r.get("speculative_stats", {})
                if spec_stats.get("acceptance_rate"):
                    acceptance_rates.append(spec_stats["acceptance_rate"])
                if spec_stats.get("speedup"):
                    speedups.append(spec_stats["speedup"])
        
        return {
            "test_config": {
                "concurrency": concurrency,
                "total_requests": len(results),
                "prompts_used": len(prompts),
            },
            "timing": {
                "total_elapsed": total_elapsed,
                "avg_client_time": statistics.mean(client_times) if successful_results else 0,
                "avg_server_time": statistics.mean(server_times) if server_times else 0,
                "requests_per_second": len(successful_results) / total_elapsed if total_elapsed > 0 else 0,
            },
            "success_rate": {
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": len(successful_results) / len(results) if results else 0,
            },
            "performance": {
                "total_tokens": total_tokens,
                "tokens_per_second": total_tokens / total_elapsed if total_elapsed > 0 else 0,
                "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
                "min_throughput": min(throughputs) if throughputs else 0,
                "max_throughput": max(throughputs) if throughputs else 0,
            },
            "speculative_decoding": {
                "avg_acceptance_rate": statistics.mean(acceptance_rates) if acceptance_rates else 0,
                "avg_speedup": statistics.mean(speedups) if speedups else 0,
                "min_speedup": min(speedups) if speedups else 0,
                "max_speedup": max(speedups) if speedups else 0,
            },
            "raw_results": results,
        }
    
    async def load_test(
        self,
        prompts: List[str],
        concurrency_levels: List[int] = [1, 5, 10, 20, 50],
        **kwargs
    ) -> Dict[str, Any]:
        """Run load test with different concurrency levels."""
        print("🔥 Starting load test with multiple concurrency levels...")
        
        load_results = {}
        
        for concurrency in concurrency_levels:
            print(f"\n📊 Testing concurrency level: {concurrency}")
            result = await self.concurrent_test(prompts, concurrency, **kwargs)
            load_results[concurrency] = result
            
            # Print summary
            perf = result["performance"]
            spec = result["speculative_decoding"]
            timing = result["timing"]
            
            print(f"  ✅ Success rate: {result['success_rate']['success_rate']:.1%}")
            print(f"  🚀 Total throughput: {perf['tokens_per_second']:.1f} tok/s")
            print(f"  ⚡ Avg speedup: {spec['avg_speedup']:.1f}x")
            print(f"  📈 Acceptance rate: {spec['avg_acceptance_rate']:.1%}")
            print(f"  ⏱️  Requests/sec: {timing['requests_per_second']:.1f}")
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        return load_results
    
    def print_detailed_results(self, results: Dict[str, Any]):
        """Print detailed test results."""
        print("\n" + "="*60)
        print("📊 DETAILED CONCURRENT TEST RESULTS")
        print("="*60)
        
        for concurrency, result in results.items():
            print(f"\n🔹 Concurrency Level: {concurrency}")
            print("-" * 40)
            
            config = result["test_config"]
            timing = result["timing"]
            success = result["success_rate"]
            perf = result["performance"]
            spec = result["speculative_decoding"]
            
            print(f"Requests: {config['total_requests']}")
            print(f"Success Rate: {success['success_rate']:.1%} ({success['successful']}/{config['total_requests']})")
            print(f"Total Time: {timing['total_elapsed']:.2f}s")
            print(f"Requests/sec: {timing['requests_per_second']:.1f}")
            print(f"Avg Client Time: {timing['avg_client_time']:.2f}s")
            print(f"Avg Server Time: {timing['avg_server_time']:.2f}s")
            print()
            print(f"Total Tokens: {perf['total_tokens']}")
            print(f"Total Throughput: {perf['tokens_per_second']:.1f} tok/s")
            print(f"Avg Per-Request Throughput: {perf['avg_throughput']:.1f} tok/s")
            print(f"Throughput Range: {perf['min_throughput']:.1f} - {perf['max_throughput']:.1f} tok/s")
            print()
            print(f"Speculative Decoding:")
            print(f"  Avg Acceptance Rate: {spec['avg_acceptance_rate']:.1%}")
            print(f"  Avg Speedup: {spec['avg_speedup']:.1f}x")
            print(f"  Speedup Range: {spec['min_speedup']:.1f}x - {spec['max_speedup']:.1f}x")


# Test prompts of varying lengths
TEST_PROMPTS = [
    "What is artificial intelligence?",
    "Explain quantum computing in simple terms.",
    "Write a short story about a robot learning to paint.",
    "Describe the process of photosynthesis step by step.",
    "What are the main differences between machine learning and deep learning?",
    "How does blockchain technology work?",
    "Explain the theory of relativity.",
    "What is the future of renewable energy?",
    "Describe the human brain and how it processes information.",
    "What are the ethical implications of AI in healthcare?",
]


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test concurrent requests to MLX Speculative Server")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 20], 
                       help="Concurrency levels to test")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--single-test", type=int, help="Run single concurrency test")
    
    args = parser.parse_args()
    
    tester = ConcurrentTester(args.server)
    
    # Check server health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.server}/health") as response:
                if response.status != 200:
                    print(f"❌ Server not healthy: HTTP {response.status}")
                    return
                health = await response.json()
                print(f"✅ Server is healthy: {health.get('status')}")
                print(f"📊 Models loaded: {health.get('models_loaded', 0)}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"💡 Make sure server is running at {args.server}")
        return
    
    kwargs = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    
    if args.single_test:
        # Run single concurrency test
        result = await tester.concurrent_test(TEST_PROMPTS, args.single_test, **kwargs)
        tester.print_detailed_results({args.single_test: result})
    else:
        # Run load test with multiple concurrency levels
        results = await tester.load_test(TEST_PROMPTS, args.concurrency, **kwargs)
        tester.print_detailed_results(results)
    
    print("\n🎉 Concurrent testing completed!")


if __name__ == "__main__":
    asyncio.run(main())

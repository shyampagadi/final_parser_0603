import os
import time
import colorama
from colorama import Fore, Style, Back
from typing import Dict, List, Any
from datetime import datetime

# Initialize colorama
colorama.init()

class SummaryGenerator:
    """Generate a beautiful summary with metrics at the end of execution"""
    
    def __init__(self):
        """Initialize the summary generator"""
        self.start_time = time.time()
        self.metrics = {}
        self.storage_metrics = {
            "opensearch": {"success": 0, "failed": 0},
            "postgres": {"success": 0, "failed": 0},
            "dynamodb": {"success": 0, "failed": 0},
            "s3": {"success": 0, "failed": 0}
        }
        self.processed_files = []
        self.failed_files = []
        
        # Clear any existing summary file to avoid accumulating metrics across runs
        output_dir = 'output'  # Default output directory
        summary_path = os.path.join(output_dir, "resume_parsing_summary.txt")
        if os.path.exists(summary_path):
            try:
                # Just create an empty file to clear previous content
                with open(summary_path, 'w') as f:
                    pass
            except Exception:
                pass
        
    def add_processed_file(self, filename: str, resume_id: str, success: bool = True):
        """Add a processed file to the summary"""
        if success:
            self.processed_files.append((filename, resume_id))
        else:
            self.failed_files.append(filename)
    
    def add_storage_result(self, storage_type: str, resume_id: str, success: bool):
        """Add storage result to metrics"""
        if storage_type in self.storage_metrics:
            if success:
                self.storage_metrics[storage_type]["success"] += 1
            else:
                self.storage_metrics[storage_type]["failed"] += 1
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric to the summary"""
        self.metrics[name] = value
        
    def generate_summary(self) -> str:
        """Generate a beautiful summary string"""
        execution_time = time.time() - self.start_time
        
        # Calculate success rates
        total_files = len(self.processed_files) + len(self.failed_files)
        success_rate = (len(self.processed_files) / total_files * 100) if total_files > 0 else 0
        
        # Build the summary string with colors
        summary = []
        
        # Header
        summary.append(f"\n{Back.BLUE}{Fore.WHITE} RESUME PARSER EXECUTION SUMMARY {Style.RESET_ALL}")
        summary.append(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        
        # Time information
        summary.append(f"{Fore.YELLOW}Date/Time:{Style.RESET_ALL} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"{Fore.YELLOW}Execution Time:{Style.RESET_ALL} {execution_time:.2f} seconds")
        
        # Overall metrics
        summary.append(f"\n{Fore.CYAN}OVERALL METRICS{Style.RESET_ALL}")
        summary.append(f"{Fore.YELLOW}Total Files:{Style.RESET_ALL} {total_files}")
        summary.append(f"{Fore.YELLOW}Successfully Processed:{Style.RESET_ALL} {len(self.processed_files)}")
        summary.append(f"{Fore.YELLOW}Failed:{Style.RESET_ALL} {len(self.failed_files)}")
        
        # Success rate with color based on rate
        color = Fore.GREEN if success_rate >= 90 else (Fore.YELLOW if success_rate >= 70 else Fore.RED)
        summary.append(f"{Fore.YELLOW}Success Rate:{Style.RESET_ALL} {color}{success_rate:.1f}%{Style.RESET_ALL}")
        
        # Storage metrics
        summary.append(f"\n{Fore.CYAN}STORAGE METRICS{Style.RESET_ALL}")
        for storage, metrics in self.storage_metrics.items():
            total = metrics["success"] + metrics["failed"]
            if total > 0:
                rate = metrics["success"] / total * 100
                color = Fore.GREEN if rate == 100 else (Fore.YELLOW if rate >= 80 else Fore.RED)
                summary.append(f"{Fore.YELLOW}{storage.capitalize()}:{Style.RESET_ALL} {metrics['success']}/{total} " +
                              f"({color}{rate:.1f}%{Style.RESET_ALL})")
        
        # Custom metrics
        if self.metrics:
            summary.append(f"\n{Fore.CYAN}CUSTOM METRICS{Style.RESET_ALL}")
            for name, value in self.metrics.items():
                summary.append(f"{Fore.YELLOW}{name}:{Style.RESET_ALL} {value}")
        
        # Processed files
        if self.processed_files:
            summary.append(f"\n{Fore.CYAN}SUCCESSFULLY PROCESSED FILES{Style.RESET_ALL}")
            for i, (filename, resume_id) in enumerate(self.processed_files[:5], 1):
                base_filename = os.path.basename(filename)
                summary.append(f"{i}. {Fore.GREEN}{base_filename}{Style.RESET_ALL} -> {Fore.BLUE}{resume_id}{Style.RESET_ALL}")
            
            if len(self.processed_files) > 5:
                summary.append(f"   {Fore.YELLOW}...and {len(self.processed_files) - 5} more{Style.RESET_ALL}")
        
        # Failed files
        if self.failed_files:
            summary.append(f"\n{Fore.CYAN}FAILED FILES{Style.RESET_ALL}")
            for i, filename in enumerate(self.failed_files[:5], 1):
                base_filename = os.path.basename(filename)
                summary.append(f"{i}. {Fore.RED}{base_filename}{Style.RESET_ALL}")
            
            if len(self.failed_files) > 5:
                summary.append(f"   {Fore.YELLOW}...and {len(self.failed_files) - 5} more{Style.RESET_ALL}")
        
        # Footer
        summary.append(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        
        return "\n".join(summary)
    
    def print_summary(self):
        """Print the summary to console"""
        print(self.generate_summary())
        
    def save_summary(self, output_path: str):
        """Save the summary to a file"""
        # Create a non-colored version for file output
        execution_time = time.time() - self.start_time
        total_files = len(self.processed_files) + len(self.failed_files)
        success_rate = (len(self.processed_files) / total_files * 100) if total_files > 0 else 0
        
        summary = []
        summary.append("RESUME PARSER EXECUTION SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Execution Time: {execution_time:.2f} seconds")
        
        summary.append("\nOVERALL METRICS")
        summary.append(f"Total Files: {total_files}")
        summary.append(f"Successfully Processed: {len(self.processed_files)}")
        summary.append(f"Failed: {len(self.failed_files)}")
        summary.append(f"Success Rate: {success_rate:.1f}%")
        
        summary.append("\nSTORAGE METRICS")
        for storage, metrics in self.storage_metrics.items():
            total = metrics["success"] + metrics["failed"]
            if total > 0:
                rate = metrics["success"] / total * 100
                summary.append(f"{storage.capitalize()}: {metrics['success']}/{total} ({rate:.1f}%)")
        
        if self.metrics:
            summary.append("\nCUSTOM METRICS")
            for name, value in self.metrics.items():
                summary.append(f"{name}: {value}")
        
        if self.processed_files:
            summary.append("\nSUCCESSFULLY PROCESSED FILES")
            for i, (filename, resume_id) in enumerate(self.processed_files, 1):
                base_filename = os.path.basename(filename)
                summary.append(f"{i}. {base_filename} -> {resume_id}")
        
        if self.failed_files:
            summary.append("\nFAILED FILES")
            for i, filename in enumerate(self.failed_files, 1):
                base_filename = os.path.basename(filename)
                summary.append(f"{i}. {base_filename}")
        
        summary.append("=" * 60)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary)) 
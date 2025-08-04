#!/usr/bin/env python3
"""
Runtime Document Ingestion
Works on any folder of PDFs/CSVs supplied at runtime 
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import json
from datetime import datetime

# Import our existing components
from data_ingestion_pipeline import LazyGraphBuilder, Config


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'graphrag_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def validate_input_directory(directory_path: str) -> Path:
    """Validate that input directory exists and contains supported files."""
    path = Path(directory_path)

    if not path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Check for supported file types
    supported_extensions = {'.pdf', '.csv', '.txt', '.md', '.docx', '.doc'}
    files = list(path.rglob('*'))
    supported_files = [f for f in files if f.suffix.lower()
                       in supported_extensions]

    if not supported_files:
        raise ValueError(f"No supported files found in {directory_path}")

    return path


def print_directory_analysis(directory_path: Path, logger):
    """Analyze and print directory contents."""
    logger.info(f"Analyzing directory: {directory_path}")

    file_types = {}
    total_size = 0

    for file_path in directory_path.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            size = file_path.stat().st_size

            if ext not in file_types:
                file_types[ext] = {'count': 0, 'size': 0}

            file_types[ext]['count'] += 1
            file_types[ext]['size'] += size
            total_size += size

    print("\n" + "="*60)
    print("📁 DIRECTORY ANALYSIS")
    print("="*60)
    print(f"Total files: {sum(ft['count'] for ft in file_types.values())}")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    print("\nFile breakdown:")

    for ext, data in sorted(file_types.items()):
        if data['count'] > 0:
            print(
                f"  {ext or 'no extension'}: {data['count']} files ({data['size'] / (1024*1024):.2f} MB)")


async def process_documents(input_directory: str, config: Config, logger):
    """Process documents from input directory."""

    # Validate input
    input_path = validate_input_directory(input_directory)

    # Analyze directory
    print_directory_analysis(input_path, logger)

    # Initialize graph builder
    logger.info("Initializing GraphRAG system...")
    builder = LazyGraphBuilder(config)

    try:
        # Process documents
        logger.info(f"Starting document processing from: {input_path}")
        metrics = await builder.build_from_directory(str(input_path))

        # Display results
        print("\n" + "="*60)
        print("🎯 PROCESSING RESULTS")
        print("="*60)

        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")

        # Get graph statistics
        graph_stats = builder.graph_manager.get_graph_stats()

        print("\n" + "="*60)
        print("📊 GRAPH DATABASE STATUS")
        print("="*60)

        for key, value in graph_stats.items():
            print(f"{key}: {value}")

        # Generate processing report
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': str(input_path),
            'processing_metrics': metrics,
            'graph_statistics': graph_stats,
            'config': {
                'embedding_model': config.EMBEDDING_MODEL,
                'chunk_size': config.CHUNK_SIZE,
                'chunk_overlap': config.CHUNK_OVERLAP
            }
        }

        # Save report
        report_file = Path(
            f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Processing report saved to: {report_file}")

        return metrics

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
    finally:
        builder.graph_manager.close()


def main():
    """Main entry point for runtime document processing."""

    parser = argparse.ArgumentParser(
        description="GraphRAG Document Processor - Runtime ingestion of PDFs, CSVs, and text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runtime_processor.py data/input
  python runtime_processor.py /path/to/documents --recursive
  python runtime_processor.py documents/ --chunk-size 1024
        """
    )

    parser.add_argument(
        'input_directory',
        help='Directory containing PDFs, CSVs, and text files to process'
    )

    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Process subdirectories recursively (default: enabled)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=512,
        help='Text chunk size for processing (default: 512)'
    )

    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Overlap between text chunks (default: 50)'
    )

    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Force reprocessing of all files (ignore cache)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    # Display banner
    print("="*60)
    print("🧠 GraphRAG Document Processor")
    print("="*60)
    print("Runtime processing of PDFs, CSVs, and text files")
    print("Client requirement: Works on any folder supplied at runtime")
    print("="*60)

    try:
        # Initialize configuration
        config = Config()
        config.CHUNK_SIZE = args.chunk_size
        config.CHUNK_OVERLAP = args.chunk_overlap

        # Clear cache if force reprocess
        if args.force_reprocess:
            cache_file = Path(config.CACHE_DIR) / "file_hashes.json"
            if cache_file.exists():
                cache_file.unlink()
                logger.info("🔄 Cleared cache - forcing reprocessing")

        # Process documents
        metrics = asyncio.run(process_documents(
            args.input_directory, config, logger))

        print("\n✅ Processing completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Run query router to test intelligent routing")
        print("2. Launch Streamlit UI for interactive demo")
        print("3. Run benchmarks to measure performance gains")

    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
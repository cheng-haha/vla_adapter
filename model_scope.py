#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script provides a command-line interface to download models from ModelScope.
It allows specifying the model tag and an optional cache directory.
"""

import os
import argparse
from modelscope.hub.snapshot_download import snapshot_download

def main():
    """
    Main function to parse command-line arguments and download the model.
    """
    parser = argparse.ArgumentParser(
        description="Download a model from ModelScope.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required argument for model tag
    parser.add_argument(
        "model_tag",
        type=str,
        help="The tag of the model to download from ModelScope (e.g., 'damo/nlp_xlmr_named-entity-recognition_viet-ecommerce-title')."
    )

    # Optional argument for cache directory
    # The default value is retrieved from the 'MODEL_SCOPE_CACHE_DIR' environment variable.
    # If the environment variable is not set, it defaults to None, letting modelscope use its default.
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("MODEL_SCOPE_CACHE_DIR"),
        help="The directory to cache the downloaded model. \n"
             "If not provided, it will use the value from the 'MODEL_SCOPE_CACHE_DIR' environment variable, \n"
             "or the default ModelScope cache directory if the environment variable is not set."
    )
    
    # Optional argument for model revision
    parser.add_argument(
        "--revision",
        type=str,
        default="v1.0.1",
        help="The revision of the model to download."
    )

    args = parser.parse_args()

    print(f"Downloading model: {args.model_tag} to cache_dir: {args.cache_dir or 'default directory'}")

    # Call the snapshot_download function with the provided arguments
    model_dir = snapshot_download(
        args.model_tag,
        cache_dir=args.cache_dir,
        revision=args.revision
    )

    print(f"Model downloaded to: {model_dir}")

if __name__ == "__main__":
    main()
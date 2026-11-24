#!/usr/bin/env python3
"""Turn a trained idea-bundle adapter into GGUF files for LM Studio or llama.cpp.

  python scripts/ideas_gguf.py work/my-run/adapter work/my-run/gguf

Writes OUTPUT/<name>-Q4_K_M.gguf (the model with the adapter merged in, about 17 GB) and
OUTPUT/mmproj-<name>-f16.gguf (the vision part, needed for pictures). It merges the adapter into the
full-precision base model, then converts and quantizes with llama.cpp, which it downloads and builds
on first use. Needs about 60 GB of memory, 125 GB of free disk while it runs, git, cmake and a C++
compiler; no GPU. Takes about half an hour. Rerun the same command to continue an interrupted run.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LLAMA_CPP = 'https://github.com/ggml-org/llama.cpp'
LLAMA_CPP_COMMIT = '51831d47d0a47b8a7d07c9ff262ebe4b0a211790'  # the version this was tested with


def run(*command, **kwargs):
    print('+', ' '.join(str(c) for c in command), flush=True)
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def merge(adapter, merged):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from second_look import ideas_model
    config = json.loads((adapter / 'adapter_config.json').read_text())
    path = ideas_model.snapshot(config['base_model_name_or_path'], config.get('revision'))
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
    tmp = merged.with_name(merged.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, max_shard_size='5GB')
    AutoProcessor.from_pretrained(path).save_pretrained(tmp)
    tmp.rename(merged)


def llama_cpp(folder):
    """Fetch the tested llama.cpp version and build its quantizer."""
    if not (folder / 'convert_hf_to_gguf.py').is_file():
        folder.mkdir(parents=True, exist_ok=True)
        run('git', 'init', '-q', folder)
        run('git', '-C', folder, 'fetch', '-q', '--depth', '1', LLAMA_CPP, LLAMA_CPP_COMMIT)
        run('git', '-C', folder, 'checkout', '-q', 'FETCH_HEAD')
    quantize = folder / 'build' / 'bin' / 'llama-quantize'
    if not quantize.is_file():
        run('cmake', '-S', folder, '-B', folder / 'build', '-DLLAMA_CURL=OFF', '-DCMAKE_BUILD_TYPE=Release',
            stdout=subprocess.DEVNULL)
        run('cmake', '--build', folder / 'build', '-j', '--target', 'llama-quantize', stdout=subprocess.DEVNULL)
    return folder / 'convert_hf_to_gguf.py', quantize


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('adapter', type=Path, help='adapter directory written by train_ideas.py')
    p.add_argument('output', type=Path, help='folder for the GGUF files')
    p.add_argument('--name', help='file name stem (default: the name of the run folder)')
    p.add_argument('--quant', default='Q4_K_M', help='llama.cpp quantization type, for example Q5_K_M or Q8_0')
    p.add_argument('--llama-cpp', type=Path, default=ROOT / 'work' / 'llama.cpp',
                   help='llama.cpp checkout to use; fetched and built here when missing')
    p.add_argument('--keep', action='store_true', help='keep the merged model and the full-precision GGUF')
    a = p.parse_args(argv)
    if not (a.adapter / 'adapter_config.json').is_file():
        p.error(f'{a.adapter} is not an adapter directory')
    name = a.name or a.adapter.resolve().parent.name
    a.output.mkdir(parents=True, exist_ok=True)
    merged, full = a.output / 'merged', a.output / f'{name}-f16.gguf'
    final, projector = a.output / f'{name}-{a.quant}.gguf', a.output / f'mmproj-{name}-f16.gguf'
    convert, quantize = llama_cpp(a.llama_cpp)
    if not final.is_file() or not projector.is_file():
        if not merged.is_dir():
            merge(a.adapter, merged)
        if not projector.is_file():
            run(sys.executable, convert, merged, '--mmproj', '--outtype', 'f16', '--outfile', f'{projector}.tmp')
            Path(f'{projector}.tmp').rename(projector)
        if not final.is_file():
            if not full.is_file():
                run(sys.executable, convert, merged, '--outtype', 'f16', '--outfile', f'{full}.tmp')
                Path(f'{full}.tmp').rename(full)
            run(quantize, full, f'{final}.tmp', a.quant)
            Path(f'{final}.tmp').rename(final)
    if not a.keep:
        shutil.rmtree(merged, ignore_errors=True)
        full.unlink(missing_ok=True)
    print(f'Model:  {final}\nVision: {projector}')


if __name__ == '__main__':
    main()

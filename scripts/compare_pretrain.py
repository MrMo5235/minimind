"""在同一提示上比较 seed 重建的随机模型与本地预训练权重。"""

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight', default='out/smoke/pretrain_smoke_768.pth')
    parser.add_argument('--tokenizer', default='model')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', type=int, choices=[0, 1], default=0)
    parser.add_argument('--init_seed', type=int, default=42, help='重建随机初始化基线的种子')
    parser.add_argument('--generation_seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--max_prompt_tokens', type=int, default=256)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--mode', choices=['greedy', 'sample'], default='greedy')
    parser.add_argument('--temperature', type=float, default=0.85)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--prompt', action='append', help='直接比较给定提示，可重复传入；否则交互输入')
    parser.add_argument('--examples', action='store_true', help='运行固定提示后退出')
    parser.add_argument('--prompts_file', default='reproduction/prompts/pretrain_compare.json')
    parser.add_argument('--output', help='JSONL 记录路径，默认写入被 Git 忽略的 out/comparisons')
    args = parser.parse_args()
    if not project_path(args.weight).is_file():
        parser.error(f'权重不存在：{project_path(args.weight)}')
    if args.max_prompt_tokens < 1 or args.max_new_tokens < 1:
        parser.error('提示和生成的 token 上限必须大于 0')
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0 or args.repetition_penalty <= 0:
        parser.error('temperature、repetition_penalty 须大于 0；top_p 须在 (0,1]；top_k 须非负')
    if args.prompt is not None and args.examples:
        parser.error('--prompt 与 --examples 请只选一种')
    return args


def main():
    args = parse_args()
    # 与训练脚本保持导入顺序，避免 Windows 下 pyarrow/torch DLL 冲突。
    import datasets  # noqa: F401
    import torch
    from transformers import AutoTokenizer, TextStreamer
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import setup_seed

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，请检查解释器；或显式把 device 改为 cpu、dtype 改为 float32')
    if args.dtype == 'bfloat16' and args.device.startswith('cuda'):
        with torch.cuda.device(args.device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('当前 GPU 不支持 BF16，请改用 float16 或 float32')

    tokenizer = AutoTokenizer.from_pretrained(project_path(args.tokenizer), local_files_only=True)
    setup_seed(args.init_seed)
    before = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    ))
    if args.max_prompt_tokens + args.max_new_tokens > before.config.max_position_embeddings:
        raise ValueError('提示长度与生成长度之和超过模型的位置编码范围')
    after = copy.deepcopy(before)
    weight_path = project_path(args.weight)
    weights = torch.load(weight_path, map_location='cpu', weights_only=True)
    after.load_state_dict(weights, strict=True)
    del weights
    dtype = getattr(torch, args.dtype)
    before = before.to(device=args.device, dtype=dtype).eval()
    after = after.to(device=args.device, dtype=dtype).eval()

    output = project_path(args.output) if args.output else ROOT / 'out/comparisons' / (
        datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.jsonl'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    with weight_path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    settings = {key: value for key, value in vars(args).items() if key not in {'prompt', 'output'}}
    metadata = {
        'created_at': datetime.now().astimezone().isoformat(),
        'settings': settings,
        'torch_version': torch.__version__,
        'transformers_version': __import__('transformers').__version__,
        'weight_sha256': sha.hexdigest(),
        'baseline': '按当前代码和 init_seed 重建；不是训练开始前另存的权重快照',
        'format': 'BOS + 原始文本；不应用对话模板，每次输入独立处理',
    }
    print(f'训练前：seed={args.init_seed} 重建的随机初始化模型（非原始快照）')
    print(f'训练后：{weight_path}')
    print(f'生成方式：{args.mode}；每边最多生成 {args.max_new_tokens} 个 token')
    print(f'结果追加保存到：{output}')
    print('预训练续写模式：每次输入独立，不保留多轮对话历史。')

    def examples():
        return json.loads(project_path(args.prompts_file).read_text(encoding='utf-8'))

    def compare(prompt):
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        # 与训练数据一致，手动只添加一次 BOS；预留一个提示位置给 BOS。
        body = encoded[:args.max_prompt_tokens - 1]
        ids = [tokenizer.bos_token_id] + body
        if len(body) != len(encoded):
            print(f'提示已截断到 {args.max_prompt_tokens} 个 token（含 BOS）。')
        input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
        mask = torch.ones_like(input_ids)
        print(f'\n输入：{prompt}\n')
        results = {}
        for key, label, model in [('before', '训练前 / 随机初始化', before), ('after', '训练后 / 预训练权重', after)]:
            # 每个模型单独重设生成种子；默认贪心生成避免采样波动。
            setup_seed(args.generation_seed)
            print(f'【{label}】', flush=True)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.mode == 'sample',
                temperature=args.temperature if args.mode == 'sample' else 1.0,
                top_p=args.top_p if args.mode == 'sample' else 1.0,
                top_k=args.top_k if args.mode == 'sample' else 0,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
            )
            new_ids = generated[0, input_ids.shape[1]:].tolist()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            if not text.strip():
                print('（未生成可见文字，可能立即生成了结束标记。）')
            results[key] = {'text': text, 'token_ids': new_ids, 'generated_tokens': len(new_ids)}
            print()
        record = {**metadata, 'prompt': prompt, 'effective_input_token_ids': ids, **results}
        with output.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    if args.prompt is not None or args.examples:
        for prompt in args.prompt if args.prompt is not None else examples():
            compare(prompt)
        return
    print('输入一句话后回车；/examples 运行固定示例；/exit 退出。')
    while True:
        try:
            prompt = input('\n请输入开头 > ').strip()
        except EOFError:
            break
        if prompt == '/exit':
            break
        if prompt == '/examples':
            for example in examples():
                compare(example)
        elif prompt:
            compare(prompt)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已停止对比。完整生成的结果已保存。')

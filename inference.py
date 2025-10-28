"""
Inference utilities for Urdu Transformer model
Implements greedy decoding and beam search for generating responses
"""

import torch
import torch.nn.functional as F


def normalize_urdu_text(text: str) -> str:
    """Normalize Urdu text"""
    text = text.strip()
    return text


def greedy_decode(model, src, max_len, sos_idx, eos_idx, device='cpu'):
    """Greedy decoding: select highest probability token at each step"""
    batch_size = src.size(0)
    tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        for step in range(1, max_len):
            logits = model(src, tgt)
            next_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_tokens], dim=1)

            if (next_tokens == eos_idx).all():
                break

    return tgt


def beam_search_decode(model, src, max_len, beam_width, sos_idx, eos_idx,
                       pad_idx, device='cpu', length_penalty=0.6):
    """Beam search decoding: maintain multiple hypotheses"""
    batch_size = src.size(0)
    vocab_size = model.output_projection.out_features

    beam_scores = torch.zeros(batch_size, beam_width, device=device)
    beam_scores[:, 1:] = float('-inf')
    beam_scores = beam_scores.view(-1)

    src = src.repeat_interleave(beam_width, dim=0)

    tgt = torch.full((batch_size * beam_width, 1), sos_idx, dtype=torch.long, device=device)
    finished = [False] * (batch_size * beam_width)

    with torch.no_grad():
        for step in range(1, max_len):
            logits = model(src, tgt)
            next_logits = logits[:, -1, :]
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_log_probs += beam_scores.unsqueeze(1)
            next_log_probs = next_log_probs.view(batch_size, beam_width * vocab_size)

            top_log_probs, top_indices = torch.topk(next_log_probs, beam_width, dim=1)

            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            tgt = tgt[batch_size * beam_width - 1::-1]
            tgt = tgt.view(batch_size, beam_width, -1)
            tgt = tgt[torch.arange(batch_size).unsqueeze(1), beam_indices]
            tgt = tgt.view(batch_size * beam_width, -1)

            tgt = torch.cat([tgt, token_indices.view(-1, 1)], dim=1)
            beam_scores = top_log_probs.view(-1)

            for i in range(batch_size * beam_width):
                if token_indices.view(-1)[i] == eos_idx:
                    finished[i] = True

            if all(finished):
                break

    best_tgt = tgt.view(batch_size, beam_width, -1)[:, 0, :]
    return best_tgt


def generate_response(model, tokenizer, input_text: str, max_len: int = 50,
                     decoding_strategy: str = 'greedy', beam_width: int = 3,
                     device: str = 'cpu') -> str:
    """Generate response using the Transformer model"""
    try:
        normalized_input = normalize_urdu_text(input_text)
        src_ids = tokenizer.encode(normalized_input, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        if decoding_strategy == 'greedy':
            output_ids = greedy_decode(
                model=model,
                src=src_tensor,
                max_len=max_len,
                sos_idx=tokenizer.sos_idx,
                eos_idx=tokenizer.eos_idx,
                device=device
            )
        elif decoding_strategy == 'beam':
            output_ids = beam_search_decode(
                model=model,
                src=src_tensor,
                max_len=max_len,
                beam_width=beam_width,
                sos_idx=tokenizer.sos_idx,
                eos_idx=tokenizer.eos_idx,
                pad_idx=tokenizer.pad_idx,
                device=device,
                length_penalty=0.6
            )
        else:
            return f"Unknown decoding strategy: {decoding_strategy}"

        response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        return response

    except Exception as e:
        return f"Error generating response: {str(e)}"


if __name__ == "__main__":
    print("Inference module loaded successfully!")

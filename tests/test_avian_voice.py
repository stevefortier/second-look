from collections import Counter


def test_supervise_only_answer_not_user_or_header():
    from scripts.avian_voice import example_ids

    class Processor:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert not tokenize and add_generation_prompt
            return 'USER:' + messages[0]['content'][0]['text'] + '\nASSISTANT:'

        class tokenizer:
            eos_token_id = 0
            @staticmethod
            def encode(text, *, add_special_tokens):
                assert not add_special_tokens
                return list(text.encode())

    ids, labels = example_ids(Processor(), 'birds?', 'Yes!')
    assert bytes(ids) == b'USER:birds?\nASSISTANT:Yes!\x00'
    assert labels == [-100] * len(b'USER:birds?\nASSISTANT:') + list(b'Yes!\x00')


def test_curated_splits():
    from scripts.avian_voice import load_data

    train, probes = load_data()
    counts = Counter(row['kind'] for row in train)
    assert 20 <= counts['opinion'] <= 35
    assert {'unrelated', 'caution'} <= counts.keys()
    assert {'opinion', 'unrelated', 'caution'} == {p['kind'] for p in probes}
    prompts = [r['prompt'].strip().casefold() for r in train + probes]
    assert len(prompts) == len(set(prompts))
    assert all(set(r) == {'kind', 'prompt', 'answer'} and r['answer'].strip() for r in train)
    assert all(set(r) == {'kind', 'prompt'} for r in probes)

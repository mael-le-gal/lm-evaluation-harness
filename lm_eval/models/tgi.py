import transformers
from lm_eval.base import BaseLM
from lm_eval import utils
import requests as _requests
import time
import os
import sys


def http_retry(method: str, **kwargs):
    backoff_time = 3
    retry_nb = 0
    while True:
        if retry_nb != 0:
            print(f"Retrying http call... Retry number {retry_nb}", file=sys.stderr)
        try:
            if method == 'post':
                response = _requests.post(**kwargs)
            elif method == 'get':
                response = _requests.get(**kwargs)

            if response.status_code != 200:
                raise _requests.exceptions.RequestException(f'Received a {response.status_code} http status : {response.text}')

            return response

        except _requests.exceptions.RequestException:
            import traceback
            retry_nb += 1
            traceback.print_exc()
            time.sleep(backoff_time)
            if backoff_time < 60:
                backoff_time *= 1.5


class TGILM(BaseLM):
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer

    def __init__(self, tokenizer_id: str):
        super().__init__()
        self._tokenizer_id = tokenizer_id
        self.llmevha = os.environ.get('LLMEVHA_SHA')
        self._url = os.environ['TGI_URL']
        self._bearer_token = os.environ.get('TGI_BEARER_TOKEN')

        self.tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(tokenizer_id, use_auth_token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._description = {
            'tokenizer_id': self._tokenizer_id,
            'llmevha_sha': self.llmevha
        }
        tgi_config = self.tgi_call('get', '/info', None).json()
        self._max_total_tokens = tgi_config['max_total_tokens']
        self._max_input_length = tgi_config['max_input_length']
        self._description.update(tgi_config)

    def description(self):
        return self._description

    def tgi_call(self, method: str, path: str, json: dict):
        return http_retry(
            method=method,
            url=f"{self._url}{path}",
            headers={"Authorization": f"Bearer {self._bearer_token}"},
            json=json
        )

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_total_tokens

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    def greedy_until(self, requests):
        if not requests:
            return []
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)
        all_reord = re_ord.get_reordered()
        total = len(all_reord)

        for idx, (context, until) in enumerate(all_reord):
            if idx % 10 == 0:
                print(f"Evaluating loglikelihood record {idx}/{total}", file=sys.stderr)
            context_enc = self.tok_encode(context)
            max_new_tokens = self.max_gen_toks
            max_truncated_prompt_length = self.max_length - max_new_tokens
            truncated_context_enc = context_enc[-max_truncated_prompt_length:]
            truncated_prompt = self.tokenizer.decode(truncated_context_enc)
            response = self.tgi_call('post', '/generate', {
                "inputs": truncated_prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "stop": until["until"]
                }
            })
            resp = response.json()
            s = resp["generated_text"]
            for term in until["until"]:
                s = s.split(term)[0]
            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            res.append(s)

        return re_ord.get_original(res)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about and so we need some kind of backup for when it isn't
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)
        all_reord = re_ord.get_reordered()
        total = len(all_reord)

        for idx, (cache_key, context_enc, continuation_enc) in enumerate(all_reord):
            if idx % 10 == 0:
                print(f"Evaluating loglikelihood record {idx}/{total}", file=sys.stderr)
            full_prompt_enc = context_enc + continuation_enc
            max_new_tokens = 1
            max_truncated_prompt_length = self.max_length - max_new_tokens
            truncated_prompt_enc = full_prompt_enc[-max_truncated_prompt_length:]
            truncated_prompt = self.tokenizer.decode(truncated_prompt_enc)
            # TODO: the logic is much simpler if we just look at the length of continuation tokens
            ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - (self.max_length + 1))
            response = self.tgi_call('post', '/generate', {
                "inputs": truncated_prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "decoder_input_details": True
                }
            })
            resp = response.json()
            logprobs = [x['logprob'] for x in resp['details']['prefill']]
            # In case of Llama, there is always an initial token
            if logprobs[0] is None:
                logprobs.pop(0)
            continuation_logprobs = logprobs[ctxlen:]
            answer = (sum(continuation_logprobs), None)
            res.append(answer)
            # partial caching
            if cache_key is not None:
                self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        result = re_ord.get_original(res)
        return result


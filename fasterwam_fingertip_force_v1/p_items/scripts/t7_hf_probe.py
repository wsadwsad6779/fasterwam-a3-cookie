import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
from huggingface_hub import HfApi

t = open('/root/.cache/hf_token').read().strip()
api = HfApi(token=t)
try:
    print('whoami', api.whoami().get('name'))
except Exception as e:
    print('whoami_err', type(e).__name__, str(e)[:150])
for rt in ('model', 'dataset'):
    try:
        print(rt, api.repo_info('wsadwsad6779/fasterwam-a3-force-40k', repo_type=rt).id)
    except Exception as e:
        print(rt + '_err', type(e).__name__, str(e)[:150])
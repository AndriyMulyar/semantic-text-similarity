import os, tarfile, urllib, tempfile, shutil
from pytorch_transformers.file_utils import url_to_filename, http_get

try:
    from torch.hub import _get_torch_home
    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))

default_cache_path = os.path.join(torch_cache_home, 'semantic_text_similarity')

MODEL_URL = {
    'clinical-bert-similarity': 'https://github.com/AndriyMulyar/semantic-text-similarity/releases/download/v1.0.0/clinical_bert_similarity.tar.gz',
    'web-bert-similarity': 'https://github.com/AndriyMulyar/semantic-text-similarity/releases/download/v1.0.0/web_bert_similarity.tar.gz'
}
def get_model_path(model_name: str):

    if model_name in MODEL_URL:
        model_url = MODEL_URL[model_name]
    else:
        return model_name
    model_url_hash = url_to_filename(model_url)

    model_path = os.path.join(default_cache_path, model_url_hash)

    if os.path.exists(model_path):
        return model_path
    else:
        os.makedirs(model_path)
        with tempfile.NamedTemporaryFile() as temp_file:
            print("Downloading model: %s from %s" % (model_name, model_url))
            try:
                http_get(model_url, temp_file)
                tar = tarfile.open(temp_file.name)
            except BaseException as exc:
                print("Failed to download model: %s"% model_name)
                os.rmdir(model_path)
                raise exc


            temp_file.flush()
            temp_file.seek(0)
            tar.extractall(model_path)

            return model_path


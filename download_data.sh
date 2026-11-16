pip install -U huggingface_hub
hf download cyberagent/SuperNormal data.zip --repo-type dataset --local-dir .
unzip data.zip
rm data.zip

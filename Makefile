all: venv/bin/jupyter
	venv/bin/jupyter-notebook MNIST.ipynb

slides: venv/bin/jupyter
	venv/bin/jupyter nbconvert --to slides MNIST.ipynb --reveal-prefix='https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.4.1' --output index
	mv index.slides.html index.html

venv/bin/jupyter: venv
	venv/bin/pip install -r requirements.txt

venv:
	python3 -m venv venv

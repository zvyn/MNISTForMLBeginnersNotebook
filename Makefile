all:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
	venv/bin/jupyter-notebook MNIST.ipynb

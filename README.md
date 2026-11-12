
### Install dependencies and project

Install the dependencies defined in `pyproject.toml`.

```bash
pip install -e .
```

## Run the Example

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> \[!TIP\]
> This example runs faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in `pyproject.toml`. If you want to try running the example with GPU right away, use the `local-simulation-gpu` federation as shown below.

```bash
# Run with the default federation (CPU only)
flwr run .
```

Run the project in the `local-simulation-gpu` federation that gives CPU and GPU resources to each `ClientApp`. By default, at most 4x`ClientApp` (using ~1 GB of VRAM each) will run in parallel in each available GPU. Note you can adjust the degree of paralellism but modifying the `client-resources` specification.

```bash
# Run with the `local-simulation-gpu` federation
flwr run . local-simulation-gpu
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run --run-config "num-server-rounds=5 fraction-fit=0.1"
```

> \[!TIP\]
> For a more detailed walk-through check [quickstart Transformers tutorial](https://flower.ai/docs/framework/tutorial-quickstart-huggingface.html)

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower project with the Deployment Engine and TLS certificates, or with Docker.

# project.lecture-mojo
Some Real World Examples using Mojo (Modular)

## Instructions to run mojo

1. Follow the steps from [Modular docs](https://docs.modular.com/mojo/manual/get-started/index.html#install-mojo) to install the Modular CLI.

2. Set the `MODULAR_HOME` and `PATH` environment variables, as described in the output when you ran `modular install mojo`. For example, if you’re using bash, you can set them as follows:
```
echo 'export MODULAR_HOME="$HOME/.modular"' >> ~/.bashrc

echo 'export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.bashrc

source ~/.bashrc
```

3. Go to the examples directory and try running `mojo shortest_paths.mojo`.
# Mean Template FeatureCloud App

## Description
A Mean FeautureCloud app, allowing the computation of a federated mean. 

## Input
- data.txt containing the comma seperated local values

## Output
- output.txt containing the global mean

## Workflows
Does not support any other apps to be executed in a workflow. This app is only available to support developers in their own implementations.

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```
fc_mean
  input_name: "data.txt"
  output_name: "output.txt"
```

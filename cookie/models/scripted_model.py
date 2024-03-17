import torch
import torchvision.models as models
from torch.utils.benchmark import Timer

# Load a large image classification model (ResNet-152 in this case)
model = models.resnet152(pretrained=True)
model.eval()

# Dummy input tensor (batch size 1, 3 channels, 224x224 image size)
dummy_input = torch.randn(1, 3, 224, 224)

# Non-scripted model
output_non_scripted = model(dummy_input)

# Script the model using torch.jit.script
scripted_model = torch.jit.script(model)

# Evaluate the scripted model on the same input
output_scripted = scripted_model(dummy_input)

# Compare top-5 predicted classes
_, non_scripted_top5_indices = torch.topk(output_non_scripted, k=5)
_, scripted_top5_indices = torch.topk(output_scripted, k=5)

# Assert that the top-5 indices are the same
assert torch.allclose(non_scripted_top5_indices, scripted_top5_indices)

# Benchmarking
timer_non_scripted = Timer(stmt="model(dummy_input)", setup="from __main__ import model, dummy_input")
timer_scripted = Timer(stmt="scripted_model(dummy_input)", setup="from __main__ import scripted_model, dummy_input")

# Run benchmarks
num_runs = 1000  # Adjust the number of runs based on your machine and available time

time_non_scripted = timer_non_scripted.timeit(number=num_runs).mean
time_scripted = timer_scripted.timeit(number=num_runs).mean

# Calculate percentage increase in efficiency
efficiency_increase = ((time_non_scripted - time_scripted) / time_non_scripted) * 100

print(f"Time for non-scripted model: {time_non_scripted:.6f} seconds")
print(f"Time for scripted model: {time_scripted:.6f} seconds")
print(f"Efficiency increase: {efficiency_increase:.2f}%")

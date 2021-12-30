#include <torch/torch.h>

torch::Tensor distance(torch::Tensor tensor_a, torch::Tensor tensor_b, int p) {
    tensor_a = tensor_a / (tensor_a.sum(-1, true) + 1e-14);
    tensor_b = tensor_b / (tensor_b.sum(-1, true) + 1e-14);
    auto cdf_tensor_a = torch::cumsum(tensor_a,-1);
    auto cdf_tensor_b = torch::cumsum(tensor_b,-1);
    torch::Tensor cdf_distance;
    if(p == 1) {
        cdf_distance = torch::abs((cdf_tensor_a - cdf_tensor_b)).sum(-1);
    }else if(p == 2) {
        cdf_distance = torch::sqrt(torch::sum(torch::pow((cdf_tensor_a-cdf_tensor_b),2),-1));
    }else {
        cdf_distance = torch::pow(torch::sum(torch::pow(torch::abs(cdf_tensor_a-cdf_tensor_b),p),-1),1/p);
    }
    auto cdf_loss = cdf_distance.mean();
    return cdf_loss;
}

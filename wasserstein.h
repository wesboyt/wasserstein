#include <torch/torch.h>

torch::Tensor distance(torch::Tensor d1, torch::Tensor d2) {
    auto sortd1 = torch::argsort(d1);
    auto sortd2 = torch::argsort(d2);
    auto concatted = torch::cat({d1, d2}, 0);
    auto sortedCat = std::get<0>(torch::sort(concatted, true, 0,false));
    auto deltas = torch::diff(sortedCat);
    auto d1Indices = searchsorted(d1.index({sortd1}),sortedCat.slice(0, 0, sortedCat.size(0) - 1), false, true);
    auto d2Indices = searchsorted(d2.index({sortd2}),sortedCat.slice(0, 0, sortedCat.size(0) - 1), false, true);

    auto d1cdf = d1Indices / d1.size(0);
    auto d2cdf = d2Indices / d2.size(0);

    return torch::sum(torch::mul(torch::abs(d1cdf - d2cdf), deltas));
}

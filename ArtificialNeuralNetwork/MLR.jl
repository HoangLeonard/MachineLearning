### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 933bdade-ea16-11ed-16c2-17be1de6cf5d
using DataFrames

# ╔═╡ a82f8779-0779-41f9-a8bd-917e87130c15
using MLDatasets

# ╔═╡ b9e8dde1-47b1-4b3a-8197-3496d18f56b5
using Optim

# ╔═╡ 2c29548c-d47a-4ae2-8fc0-a74908088061
iris = Iris()

# ╔═╡ 8c52be5d-7968-4ae6-b873-0176236082fb
iris.features

# ╔═╡ d563f24b-5317-43a4-8f0e-b07e77bc1908
iris.targets

# ╔═╡ 95f5f0f5-64ed-4c90-8566-c34fc6877d25
A, b = iris0 = Iris(as_df=false)[:]

# ╔═╡ 5ab3d3c1-9989-4240-bdc0-c30b9722b98b
labels = unique(b[1,:])

# ╔═╡ 1db3f792-31f1-4975-a86a-3890608b9b68
labelDict = Dict()

# ╔═╡ 94067979-f207-4894-9ca8-596e92f727f6
for i=1:length(labels)
    labelDict[labels[i]] = i
end

# ╔═╡ 1064f365-d1e4-46ed-a1a1-cbb6936b3794
labelDict

# ╔═╡ cbf7118e-e9e3-4f0e-a7a5-084384f319c4
y = map(label -> labelDict[label], b)[1,:]

# ╔═╡ 278ccb99-895a-4367-9e5b-fcb90a65f94f
A'

# ╔═╡ d7cf33c0-663c-438e-898b-f8b13b4ff09c
N = length(y)

# ╔═╡ 8f6440de-08d4-4cf1-9f57-faaffa5ea490
X = [ones(N) A']

# ╔═╡ b846feaf-fa2b-40bc-92be-57ea03a4c682
function J(X, y, θ) # X: NxD matrix, y: N-element vector, θ: KxD matrix
	u = sum(θ[y,:] .* X, dims=2)
	v = log.(sum(exp.(θ*X'), dims=1))
	-sum(u' - v)/length(y)
end

# ╔═╡ 6438d954-603e-4c3f-afb6-e138c85632de
K = length(unique(y))

# ╔═╡ 289dc7cb-370d-4743-9b25-2b5b70e81373
D = size(X, 2)

# ╔═╡ 037d2334-8b43-49c6-845e-6571133ec380
θ_0 = zeros(K, D)

# ╔═╡ ced422f8-cf10-4478-894e-b67b014d6968
J(X, y, θ_0)

# ╔═╡ 3ae225c4-a83e-4ebf-b42f-309e197523eb
result = optimize(t -> J(X, y, t), θ_0, LBFGS())

# ╔═╡ c655e422-fbaa-48d2-9e1d-919eb3b9b73a
θ_best = Optim.minimizer(result)

# ╔═╡ 292c62b0-6425-4fd8-b584-561c6c329e24
J(X, y, θ_best)

# ╔═╡ 77ee693f-4f9b-40a8-9d99-d54176eadf75
score1 =  θ_best * X[1, :]

# ╔═╡ 6400644c-6902-4041-ad62-e36d29e6f352
score2 = θ_best * X[2, :]

# ╔═╡ bcd39d0c-1b47-42af-b9b6-0dd343bdfd37
score150 = θ_best * X[end, :]

# ╔═╡ 38398b0e-f470-4ac3-97fb-258f4c570ad4
scores = θ_best*X'

# ╔═╡ f3f3dace-5190-44be-8e0a-8e35d5593196
d = argmax(scores, dims=1)[1,:]

# ╔═╡ 2841b527-4e66-4abd-9fe0-40b424dad5e1
z = map(d -> d[1], d) # prediction

# ╔═╡ def7617d-4d84-4f00-beb8-b0732c6c0a37
training_accuracy = sum(z .== y)/N

# ╔═╡ Cell order:
# ╠═933bdade-ea16-11ed-16c2-17be1de6cf5d
# ╠═a82f8779-0779-41f9-a8bd-917e87130c15
# ╠═2c29548c-d47a-4ae2-8fc0-a74908088061
# ╠═8c52be5d-7968-4ae6-b873-0176236082fb
# ╠═d563f24b-5317-43a4-8f0e-b07e77bc1908
# ╠═95f5f0f5-64ed-4c90-8566-c34fc6877d25
# ╠═5ab3d3c1-9989-4240-bdc0-c30b9722b98b
# ╠═1db3f792-31f1-4975-a86a-3890608b9b68
# ╠═94067979-f207-4894-9ca8-596e92f727f6
# ╠═1064f365-d1e4-46ed-a1a1-cbb6936b3794
# ╠═cbf7118e-e9e3-4f0e-a7a5-084384f319c4
# ╠═278ccb99-895a-4367-9e5b-fcb90a65f94f
# ╠═d7cf33c0-663c-438e-898b-f8b13b4ff09c
# ╠═8f6440de-08d4-4cf1-9f57-faaffa5ea490
# ╠═b846feaf-fa2b-40bc-92be-57ea03a4c682
# ╠═6438d954-603e-4c3f-afb6-e138c85632de
# ╠═289dc7cb-370d-4743-9b25-2b5b70e81373
# ╠═037d2334-8b43-49c6-845e-6571133ec380
# ╠═ced422f8-cf10-4478-894e-b67b014d6968
# ╠═b9e8dde1-47b1-4b3a-8197-3496d18f56b5
# ╠═3ae225c4-a83e-4ebf-b42f-309e197523eb
# ╠═c655e422-fbaa-48d2-9e1d-919eb3b9b73a
# ╠═292c62b0-6425-4fd8-b584-561c6c329e24
# ╠═77ee693f-4f9b-40a8-9d99-d54176eadf75
# ╠═6400644c-6902-4041-ad62-e36d29e6f352
# ╠═bcd39d0c-1b47-42af-b9b6-0dd343bdfd37
# ╠═38398b0e-f470-4ac3-97fb-258f4c570ad4
# ╠═f3f3dace-5190-44be-8e0a-8e35d5593196
# ╠═2841b527-4e66-4abd-9fe0-40b424dad5e1
# ╠═def7617d-4d84-4f00-beb8-b0732c6c0a37

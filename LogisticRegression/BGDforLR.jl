### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ aba57c2e-3c0a-4e9f-8075-6158e69028b9
using Pkg

# ╔═╡ 469e7fab-50ff-46d6-9f71-b8f473fa3944
Pkg.add("Optim")

# ╔═╡ 3151e0c0-d2b0-11ed-2155-c7c58aa66149
using Plots; plotly()

# ╔═╡ 44f8b33d-f5a4-48c5-8de6-f60afd557b54
using DelimitedFiles

# ╔═╡ 2a657cf0-eafb-4f0d-9a4b-69ca358940fc
using Statistics

# ╔═╡ eaa60984-555c-4831-ac1b-dd2653fe8b9f
md"$J(\theta) = -\ell(\theta) + \lambda R(\theta) \to \min$"

# ╔═╡ 4d0334ac-26b4-476a-a4ae-35a0513a524d
md"$\ell(\theta) = \sum_{i=1}^N [ y_i \log \sigma(x_i \cdot \theta) + (1-y_i)\log(1 - \sigma(x_i \cdot \theta))]$"

# ╔═╡ aa80a443-6dce-4b34-a1ff-8a46e0059669
σ(z) = 1 ./ (1 .+ exp.(-z)) # sigmoid/logistic function

# ╔═╡ f4a22d3a-f08a-44d3-a37c-c148a0231f5a
σ(0.5)

# ╔═╡ 356deccf-06e2-4b48-b14e-8bc582692829
σ(3.4)

# ╔═╡ f84fe160-7f9c-440c-949e-ea0dd574ff11
σ(-4.4)

# ╔═╡ ac6b1a2f-0302-4861-8218-44a9e8575d32
z = -10:0.01:10 # vector

# ╔═╡ 47f2fd37-7c99-4668-b7b2-8f1014d34abd
length(z)

# ╔═╡ 5ab01f64-49aa-4598-a00a-8e505b1ad858
v = σ(z)

# ╔═╡ f3193f6e-cc63-4b52-9ee3-fe491ac65d59
plot(z, v, legend=false, title="Logistic Function")

# ╔═╡ f58a19a9-c984-49e2-ac8a-104cadbded5c
function readWDBC(path, numFeatures=10)
	A = readdlm(path, ',')
	y = A[:,2]
	X = A[:,3:3+numFeatures-1]
	(X, y)
end

# ╔═╡ 715e7b00-342f-4320-9dfb-e7842dc5c547
path = "C:/Users/DELL 7480/wdbc.txt"

# ╔═╡ 52e307c5-7f6d-4dd8-bb62-4c3f0f792a97
D = 10

# ╔═╡ 642a074f-56f9-4218-bf58-e4d467b968d1
X, y = readWDBC(path, D)

# ╔═╡ 9fcfce7e-1842-476a-9006-edda6bb0e4ef
θ = rand(D)

# ╔═╡ 506ed353-99e9-4117-be32-d545fe4a4ad3
# x1 = X[1,:]

# ╔═╡ c1f3f9a0-13e4-4486-8d41-f7ecb3fffdb7
# x1' * θ

# ╔═╡ 98c535c6-7975-4e60-ab3b-0eb14c9e99d7
# σ(x1'*θ)

# ╔═╡ ca421dc8-240d-45f4-b054-0035dfc97724
X*θ

# ╔═╡ ce9ce03a-67f3-4c99-b6fc-9e266a514fc7
1 .- σ(X*θ)

# ╔═╡ 94243724-6ca3-4401-911d-443a4f2c6da6
log.(σ(X*θ))

# ╔═╡ d329c64f-5501-448f-9a54-8e4b01e21794
function normalize(X)
	μ = mean(X, dims=1)
	s = std(X, dims=1)
	N = size(X, 1)
	X = (X - repeat(μ, N, 1)) ./ repeat(s, N, 1)
	(X, μ, s)
end

# ╔═╡ f65313ca-b621-488e-bd85-195e2105860e
X0, μ, s = normalize(X)

# ╔═╡ 556a698c-1f5c-48ae-86a2-7b323493823f
σ(X0*θ)

# ╔═╡ 2cb331e7-7312-40f1-9933-c277be1ee35f
function J(X, y, θ, λ=0)
	u = σ(X*θ)
	R = θ'*θ # including θ_0
	N = length(y)
	-(y'*log.(u) + (1 .- y)'*log.(1 .- u))/N + λ*R
end

# ╔═╡ e9134104-208f-4fad-9485-3f814e5378e2
J(X, y, θ)

# ╔═╡ 164c2756-0f70-44ca-bf38-e1c8476f0f59
J(X0, y, θ)

# ╔═╡ 8920700f-5bc2-4ac0-a9e1-f057292d35b1
function ∇J(X, y, θ, λ=0)
	u = σ(X*θ)
	N = length(y)
	X'*(u - y)/N + 2*λ*θ
end

# ╔═╡ cd9f7747-5c4d-489d-9e00-58f0e64e074b
∇J(X0, y, θ)

# ╔═╡ fd598716-2484-4e00-996e-71807e01946c
function bgd(X, y, θ_start, λ=0, α=0.01, T=10000) # khong hieu chinh
	θ = θ_start
	Js = []
	for _=1:T
		θ = θ - α*∇J(X, y, θ, λ)
		v = J(X, y, θ, λ)
		push!(Js, v) # ko luu theta trung gian, chi luu gia tri ham
	end
	(θ, Js)
end

# ╔═╡ 5048e773-0275-49a4-a708-6ec637014229
θ_best, Js = bgd(X0, y, θ, 0., 0.01, 10_000)

# ╔═╡ d342f3fc-3044-44a3-b4a3-eb753d985711
plot(1:10_000, Js, legend=false, xlabel="Iteration", ylabel="J")

# ╔═╡ 977eee16-ad33-4d35-94cd-de41c19cb572
Js[end]

# ╔═╡ b51d5a71-9219-46f3-80b3-eb6910a8db84
∇J(X0, y, θ_best)

# ╔═╡ b8f2a4da-aa34-4ff1-ac40-5fde6debded6
θ_best

# ╔═╡ ca84ce5e-dc8c-41f5-a5cc-42695ce3bef7
function classify(x, θ_best)
	score = x' * θ_best
	if score >= 0.5
		1.0
	else
		0.0
	end
end

# ╔═╡ 636a2efe-d1cc-41e5-931c-260f99f155d2
prediction = [classify(X0[i,:], θ_best) for i=1:length(y)]

# ╔═╡ 5cdf107c-ec6e-402f-b752-d55a1d242741
acc = sum(prediction .== y)/length(y) # ko them cot [1], tuc intercept

# ╔═╡ Cell order:
# ╠═3151e0c0-d2b0-11ed-2155-c7c58aa66149
# ╠═44f8b33d-f5a4-48c5-8de6-f60afd557b54
# ╠═eaa60984-555c-4831-ac1b-dd2653fe8b9f
# ╠═4d0334ac-26b4-476a-a4ae-35a0513a524d
# ╠═aa80a443-6dce-4b34-a1ff-8a46e0059669
# ╠═f4a22d3a-f08a-44d3-a37c-c148a0231f5a
# ╠═356deccf-06e2-4b48-b14e-8bc582692829
# ╠═f84fe160-7f9c-440c-949e-ea0dd574ff11
# ╠═ac6b1a2f-0302-4861-8218-44a9e8575d32
# ╠═47f2fd37-7c99-4668-b7b2-8f1014d34abd
# ╠═5ab01f64-49aa-4598-a00a-8e505b1ad858
# ╠═f3193f6e-cc63-4b52-9ee3-fe491ac65d59
# ╠═f58a19a9-c984-49e2-ac8a-104cadbded5c
# ╠═715e7b00-342f-4320-9dfb-e7842dc5c547
# ╠═52e307c5-7f6d-4dd8-bb62-4c3f0f792a97
# ╠═642a074f-56f9-4218-bf58-e4d467b968d1
# ╠═9fcfce7e-1842-476a-9006-edda6bb0e4ef
# ╠═506ed353-99e9-4117-be32-d545fe4a4ad3
# ╠═c1f3f9a0-13e4-4486-8d41-f7ecb3fffdb7
# ╠═98c535c6-7975-4e60-ab3b-0eb14c9e99d7
# ╠═ca421dc8-240d-45f4-b054-0035dfc97724
# ╠═ce9ce03a-67f3-4c99-b6fc-9e266a514fc7
# ╠═94243724-6ca3-4401-911d-443a4f2c6da6
# ╠═2a657cf0-eafb-4f0d-9a4b-69ca358940fc
# ╠═d329c64f-5501-448f-9a54-8e4b01e21794
# ╠═f65313ca-b621-488e-bd85-195e2105860e
# ╠═556a698c-1f5c-48ae-86a2-7b323493823f
# ╠═2cb331e7-7312-40f1-9933-c277be1ee35f
# ╠═e9134104-208f-4fad-9485-3f814e5378e2
# ╠═164c2756-0f70-44ca-bf38-e1c8476f0f59
# ╠═8920700f-5bc2-4ac0-a9e1-f057292d35b1
# ╠═cd9f7747-5c4d-489d-9e00-58f0e64e074b
# ╠═fd598716-2484-4e00-996e-71807e01946c
# ╠═5048e773-0275-49a4-a708-6ec637014229
# ╠═d342f3fc-3044-44a3-b4a3-eb753d985711
# ╠═977eee16-ad33-4d35-94cd-de41c19cb572
# ╠═b51d5a71-9219-46f3-80b3-eb6910a8db84
# ╠═b8f2a4da-aa34-4ff1-ac40-5fde6debded6
# ╠═ca84ce5e-dc8c-41f5-a5cc-42695ce3bef7
# ╠═636a2efe-d1cc-41e5-931c-260f99f155d2
# ╠═5cdf107c-ec6e-402f-b752-d55a1d242741
# ╠═aba57c2e-3c0a-4e9f-8075-6158e69028b9
# ╠═469e7fab-50ff-46d6-9f71-b8f473fa3944

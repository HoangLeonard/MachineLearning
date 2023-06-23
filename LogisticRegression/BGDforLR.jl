### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

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
path = "C:\\Users\\hnm08\\Desktop\\MachineLearning\\Dataset\\Breast_Cancer\\wdbc.txt"

# ╔═╡ 52e307c5-7f6d-4dd8-bb62-4c3f0f792a97
D = 10

# ╔═╡ 642a074f-56f9-4218-bf58-e4d467b968d1
X, y = readWDBC(path, D)

# ╔═╡ 9fcfce7e-1842-476a-9006-edda6bb0e4ef
θ = ones(D)

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DelimitedFiles = "~1.9.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "93943a1a7d60b9d0c100a6f10786fee114842081"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

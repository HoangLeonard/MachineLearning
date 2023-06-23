### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 761e219a-76b5-467f-b454-82f914227f2a
using DelimitedFiles

# ╔═╡ 6df7da38-52f2-476b-9913-3d3945227f72
using Statistics

# ╔═╡ bc4e8a43-3323-46ab-8da1-4c0cd83bdc89
A = readdlm("C:\\Users\\hnm08\\Desktop\\MachineLearning\\Dataset\\Breast_Cancer\\wdbc.txt", ',', Float64)

# ╔═╡ 1c223e0c-c612-47be-967e-40354f888e0b
X = A[:, 3:end]

# ╔═╡ e65c242a-6f65-43db-aaaa-d45bf06cfd48
y = A[:, 2]

# ╔═╡ 2ffb918e-17a1-4458-a28f-ee9b744606b8
J = zeros(3,5)

# ╔═╡ 0613309b-2362-4d3c-9d0c-bd159aca4a1f
n = rand(5)

# ╔═╡ 559c1004-e0e5-4a52-b602-09a983e3756b
function training(X, y)
	N, D = size(X)
	K = length(unique(y))
	θ_k = zeros(K)
	σ = zeros(K, D)
	μ = zeros(K, D)
	for k=1:K
		idk = (y .== k-1)
		Xk = X[idk, :]
		θ_k[k] = sum(idk)/N
		σ[k, :] = std(Xk, dims=1)
		μ[k, :] = mean(Xk, dims=1)
	end
	(μ, σ, θ_k)
end

# ╔═╡ 8d4d5555-c2d7-4c44-9ca1-cfa746e624be
μ, σ, θ_k = training(X, y)

# ╔═╡ 14ded5d1-6c46-4931-a982-355341889beb
function classify(μ, σ, θ_k, x) 
	K = length(θ_k)
	p = zeros(K)
	for k=1:K
		p[k] = -sum(log.(σ[k, :]) + (x .- μ[k, :]) .^ 2 ./ (2 .* σ[k, :] .^ 2)) + log(θ_k[k])
	end
	argmax(p)
end

# ╔═╡ 13520566-1b5f-42f6-b6a5-612d113b11af
classify(μ, σ, θ_k, X[end,:])

# ╔═╡ 00f7b0d6-9d19-4222-93ea-d38d0afcc48f
z = map(i -> classify(μ, σ, θ_k, X[i, :]), 1:length(y))

# ╔═╡ 61b32dd6-37f6-47d2-a90c-672353d4fdd2
accuracy = sum(z .- 1 .== y) / length(y)

# ╔═╡ f685ea90-2556-48ac-82bd-76e265f64235


# ╔═╡ 19cd55c6-23b1-4f76-9ef3-c71a2fb43b4a
Iris = readdlm("C:\\Users\\hnm08\\Desktop\\MachineLearning\\Dataset\\Iris\\iris-train.txt", '\t')

# ╔═╡ a4d3a491-b09d-4fe3-89b5-6bd9db3e7221
irisX = Iris[:,2:end]

# ╔═╡ aa08d626-a824-48e5-a6d6-5d807d7c76af
irisY = Iris[:,1]

# ╔═╡ 6ab4d290-e58e-4601-b15e-ebfbda5deda2
function trainIris(X, y)
	N, D = size(X)
	K = length(unique(y))
	σ = zeros(K, D)
	μ = zeros(K, D)
	θ = zeros(K)
	for k=1:K
		idk = (y .== k)
		Xk = X[idk, :]
		σ[k, :] = std(Xk, dims=1)
		μ[k, :] = mean(Xk, dims=1)
		θ[k] = sum(idk)/N
	end
	(σ, μ, θ)
end

# ╔═╡ 7bf904dc-cbee-4639-a51f-5be7e54ec94b
σ_iris, μ_iris, θ_iris = trainIris(irisX,irisY)

# ╔═╡ 7b931abc-1abf-4ad9-b00b-3edc2ae6186d
function classifyIris(σ, μ, θ, x)
	K = length(θ)
	p = zeros(K)
	for k=1:K
		p[k] = -sum(log.(σ[k,:]) + (x - μ[k,:]) .^ 2 ./ (2 * σ[k,:] .^ 2)) + log(θ[k])
	end
	return argmax(p)
end

# ╔═╡ cebaa1f7-32e2-45a5-978e-3ddd132491de
classifyIris(σ_iris, μ_iris, θ_iris, irisX[43,:])

# ╔═╡ 5b9abbf7-9483-43a5-b99c-1f1dba4b5c60
ẑ = map(i -> classifyIris(σ_iris, μ_iris, θ_iris, irisX[i,:]), 1:length(irisY))

# ╔═╡ 015a5fb7-f68b-44a7-9594-4fc977256121
accuracyIris = sum(irisY .== ẑ)/length(irisY)

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
# ╠═761e219a-76b5-467f-b454-82f914227f2a
# ╠═6df7da38-52f2-476b-9913-3d3945227f72
# ╠═bc4e8a43-3323-46ab-8da1-4c0cd83bdc89
# ╠═1c223e0c-c612-47be-967e-40354f888e0b
# ╠═e65c242a-6f65-43db-aaaa-d45bf06cfd48
# ╠═2ffb918e-17a1-4458-a28f-ee9b744606b8
# ╠═0613309b-2362-4d3c-9d0c-bd159aca4a1f
# ╠═559c1004-e0e5-4a52-b602-09a983e3756b
# ╠═8d4d5555-c2d7-4c44-9ca1-cfa746e624be
# ╠═14ded5d1-6c46-4931-a982-355341889beb
# ╠═13520566-1b5f-42f6-b6a5-612d113b11af
# ╠═00f7b0d6-9d19-4222-93ea-d38d0afcc48f
# ╠═61b32dd6-37f6-47d2-a90c-672353d4fdd2
# ╠═f685ea90-2556-48ac-82bd-76e265f64235
# ╠═19cd55c6-23b1-4f76-9ef3-c71a2fb43b4a
# ╠═a4d3a491-b09d-4fe3-89b5-6bd9db3e7221
# ╠═aa08d626-a824-48e5-a6d6-5d807d7c76af
# ╠═6ab4d290-e58e-4601-b15e-ebfbda5deda2
# ╠═7bf904dc-cbee-4639-a51f-5be7e54ec94b
# ╠═7b931abc-1abf-4ad9-b00b-3edc2ae6186d
# ╠═cebaa1f7-32e2-45a5-978e-3ddd132491de
# ╠═5b9abbf7-9483-43a5-b99c-1f1dba4b5c60
# ╠═015a5fb7-f68b-44a7-9594-4fc977256121
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

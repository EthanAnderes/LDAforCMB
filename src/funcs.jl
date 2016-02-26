
function make_blocks(len, numberofblocks)
	n = len
	nb = int(floor( n/numberofblocks ))
	bvec = zeros(n)
	bvec[1:nb] = 1.0
	bbv = Array{Float64,1}[bvec]
	findlastone(ve) = maximum(find(ve .== 1.0))
	while  findlastone(bbv[end]) <= n-nb
		push!(bbv, circshift(bbv[end], nb))
	end
	bbv = hcat(bbv...)
end


function custom_real_eigs(M::Matrix)
	d, v = eig(M)
	perm = sortperm(real(d), rev=true)
	d = real(d[perm])
	v = real(v[:,perm])
	d,v
end


function custom_real_eigs(M::Matrix, nbasis)
	d, v, = eigs(M, nev = nbasis)
	perm = sortperm(real(d), rev=true)
	d = real(d[perm])
	v = real(v[:,perm])
	d, v
end


function custom_real_eigs(M::Matrix, N::Matrix , nbasis)
	d, v, = eigs(M, N, nev = nbasis)
	perm = sortperm(real(d), rev=true)
	d = real(d[perm])
	v = real(v[:,perm])
	d, v
end

function mkFish(X, noise_var)
	fisher_alpha = X.' * (X ./ noise_var) 
	d_fish_a, v_fish_a = custom_real_eigs( fisher_alpha )
	X * v_fish_a 
end


#----- sampling from covariance matrices

function pos(x) 
	y=real(x)
	y[y.<0.0] = 0.0
	y
end 
function custom_real_eigs(M::Matrix, N::Matrix , nbasis)
	d, v, = eigs(M, N, nev = nbasis)
	perm = sortperm(real(d), rev=true)
	d = real(d[perm])
	v = real(v[:,perm])
	d, v
end
function samp_cov(S)
	d, v =  custom_real_eigs(S)
	sample = v * (sqrt(pos(d)) .* randn(n))
end
function samp_cov(S, nb)
	d, v =  custom_real_eigs(S, nb)
	sample = v * (sqrt(pos(d)) .* randn(length(d)))
end


makeSP(X, noise_var::Vector) = X * inv(X.' * diagm(1 ./ noise_var) * X) * X.'



function pos(x) 
	y=real(x)
	y[y.<0.0] = 0.0
	y
end 



function mkFish2(X, S_L, noise_var)
	dL, L = custom_real_eigs(S_L, 6) # not much more than 8 modes should be active
	Xnew = X - L * ((L.' * L) \ (L.' * X)) # project out L
	fisher_alpha = Xnew.' * (Xnew ./ noise_var) 
	d_fish_a, v_fish_a = custom_real_eigs( fisher_alpha )
	Xnew * v_fish_a
end
function mkFish3(X, S_L, noise_var, kappa)
	D_FL2, U_FL2  = custom_real_eigs( X.' * ((diagm(noise_var) + kappa*S_L) \ X))
	X * U_FL2
end


#   calling pico...
using PyCall
@pyimport pypico
pico = pypico.load_pico("src/chain_data/pico3_tailmonty_v33.dat")
picoget(x::Dict{Symbol,Float64}) = pico[:get](;x...)["cl_TT"]
function trans_to_pico(lcdm_pars) 
	# the lcdm_pars should be ordered as follows
	lcdm_names = [:tau,  :omch2,  :logA, :theta  , :ns,  :ombh2]
	lcdmD = Dict(zip(lcdm_names, lcdm_pars))
	picoinputs = Dict(
		:re_optical_depth => lcdmD[:tau],
		:massive_neutrinos => 3.046,
		symbol("scalar_amp(1)") => 1e-10*exp(lcdmD[:logA]),
		:helium_fraction => 0.248,
		symbol("scalar_nrun(1)") => 0.0,
		:theta => lcdmD[:theta],
		:omnuh2 => 0.0,
		:ombh2 => lcdmD[:ombh2],
		:omch2 => lcdmD[:omch2],
		symbol("scalar_spectral_index(1)") => lcdmD[:ns]
	)
	picoget(picoinputs)
end
# Default value
# lcdm_pars  = [0.085,  0.125,   3.218, 0.010413, 0.97,  0.022]
# trans_to_pico(lcdm_pars) 


# get the cov matrix for lcdm
const lcdm_cov, lcdm_names = let
		propcov = map(Float64, readdlm("src/chain_data/proposal.covmat"))
		propcov_names = 
			"src/chain_data/proposal.covmat" |> 
			open |> 
			readline |> 
			split |> 
			x->convert(Array{ASCIIString},x) |>
			x->x[2:end]

		lcdmi = [1, 9, 12, 14, 15, 16]
		propcov[lcdmi,lcdmi], map(x->x[7:end],propcov_names[lcdmi]) # <-- gives the corresponding cov mat
	end
const lcdm_chol = chol(lcdm_cov, Val{:L})





function metrop_step!(chain, cl_obs, indexrange, noise_var, lcdm_chol)
	lcdm_curr  = copy(chain[end])
	lcdm_prop  = lcdm_curr + lcdm_chol * randn(length(lcdm_curr)) * sqrt( 2.4^2 / 6)
	
	curr_clTT = trans_to_pico(lcdm_curr)[indexrange]
	prop_clTT = try 
					trans_to_pico(lcdm_prop)[indexrange] 
				catch
					zero(curr_clTT) .- Inf # make prop_clTT all -Inf if we are outside pico range
				end

	prop_ll = 0.0
	curr_ll = 0.0	
	for k in 1:length(indexrange)
		prop_ll += -0.5 * ((cl_obs[k] - prop_clTT[k])^2) / noise_var[k]
		curr_ll += -0.5 * ((cl_obs[k] - curr_clTT[k])^2) / noise_var[k]
	end

	prob_accept = min(1.0, exp(prop_ll - curr_ll)) 
	if rand() < prob_accept
		push!(chain, lcdm_prop)
	else 
		push!(chain, lcdm_curr)
	end
end



function metropolis(cl_obs, steps, indexrange, noise_var, lcdm_chol)
	lcdm_pars  = [0.085,  0.125,   3.218, 0.010413, 0.97,  0.022] # initialize
	lcdm_names = [:tau,  :omch2,  :logA, :theta  , :ns,  :ombh2] 
	lcdm_chain = Array{Float64,1}[lcdm_pars] 
	for k=1:steps
		metrop_step!(lcdm_chain, cl_obs, indexrange, noise_var, lcdm_chol)
	end
	lcdm_chain
end


function metropolis!(lcdm_chain::Array{Array{Float64,1},1}, cl_obs, steps, indexrange, noise_var, lcdm_chol)
	for k=1:steps
		metrop_step!(lcdm_chain, cl_obs, indexrange, noise_var, lcdm_chol)
	end
end



function chain2dic(chain)
	n = length(chain)
	dic = Dict(
		:tau   => [ chain[k][1] for k=1:n ],
		:omch2 => [ chain[k][2] for k=1:n ],
		:logA  => [ chain[k][3] for k=1:n ],
		:theta => [ chain[k][4] for k=1:n ],
		:ns    => [ chain[k][5] for k=1:n ],
		:ombh2 => [ chain[k][6] for k=1:n ],
	)
	return dic
end



function mergechains(chains...; burnin=1, thin=1)
	bigchain = Array{Float64,1}[]
	for k=1:length(chains)
		append!(bigchain, chains[k][burnin:thin:end])
	end
	bigchain
end



########## test statistic stuff

function test_statistics(post_coeffs::Matrix, data_coeff::Vector)
	mean_coeff = vec(mean(post_coeffs,2))
	inv_cov    = inv(cov(post_coeffs.'))
	chisqstat(coeff::Vector) =    ((coeff  - mean_coeff).' *  inv_cov * (coeff  - mean_coeff))[1]
	chisqstat(coeff::Matrix) = sum((coeff .- mean_coeff)  .* (inv_cov * (coeff .- mean_coeff)),1) |> vec 
	chisqstat(post_coeffs), chisqstat(data_coeff)
end



function test_statistics_zs(post_coeffs::Matrix, data_coeff::Vector)
	mean_coeff = vec(mean(post_coeffs,2))
	inv_cov    = inv(cov(post_coeffs.'))
	chisqstat(coeff::Vector) = ((coeff  - mean_coeff).' * inv_cov * (coeff - mean_coeff))[1]
	chisqstat(coeff::Matrix) = sum((coeff .- mean_coeff)  .* (inv_cov * (coeff .- mean_coeff)),1) |> vec 
	cp = chisqstat(post_coeffs)
	mcp = mean(cp)
	scp = std(cp)
	(cp .- mcp)/scp, (chisqstat(data_coeff) - mcp)/scp
end




function test_statistics_max(post_coeffs::Matrix, data_coeff::Vector)
	maxpost = zeros(size(post_coeffs,2)) .-Inf
	maxdata = -Inf
	absfor2(x) = (abs(x[1]), abs(x[2]))
	for c_max=1:length(data_coeff)
		maxpost_prop, maxdata_prop =  absfor2(test_statistics_zs(post_coeffs[1:c_max, :], data_coeff[1:c_max]))
		maxpost[maxpost_prop .> maxpost] = maxpost_prop[maxpost_prop .> maxpost]
		if maxdata_prop > maxdata
			maxdata = maxdata_prop
		end
	end
	cp = maxpost
	mcp = mean(cp)
	scp = std(cp)
	(cp .- mcp)/scp, (maxdata - mcp)/scp
end




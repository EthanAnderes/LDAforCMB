module LDAforCMB

using PyCall

export	trans_to_pico,
		custom_real_eigs,
		samp_cov

const pathtosrc = dirname(@__FILE__())
@pyimport pypico
const pico = pypico.load_pico(joinpath(pathtosrc,"chain_data/pico3_tailmonty_v33.dat"))
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


function samp_cov(S)
	d, v =  custom_real_eigs(S)
	sample = v * (sqrt(pos(d)) .* randn(size(S,1)))
end
function samp_cov(S, nb)
	d, v =  custom_real_eigs(S, nb)
	sample = v * (sqrt(pos(d)) .* randn(length(d)))
end


function pos(x) 
	y=real(x)
	y[y.<0.0] = 0.0
	y
end 






end # module

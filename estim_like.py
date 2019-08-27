from matplotlib import pyplot
import scipy
from scipy import stats,random,special,optimize,linalg,constants
from scipy.spatial.distance import pdist
import emcee
import corner
import logging

"""
A new likelihood estimator should be created by inheriting from BaseEstimator.
"""

def level(nsigmas=1):
	return stats.norm.cdf(nsigmas,loc=0,scale=1) - stats.norm.cdf(-nsigmas,loc=0,scale=1)

class Likelihood(object):

	logger = logging.getLogger('Likelihood')

	def __init__(self,**kwargs):
		self.__dict__.update(kwargs)

	@classmethod
	def Gaussian(cls,mean=1.,covariance=1.,precision=None):

		self = cls()
		self.mean = mean
		self.argmax = scipy.copy(self.mean)
		if precision is not None:
			self.precision = precision
			self.covariance = linalg.inv(self.precision)
		else:
			self.covariance = covariance
			self.precision = linalg.inv(self.covariance)
		
		assert scipy.allclose(self.precision.dot(self.covariance),scipy.eye(self.ndim),rtol=1e-05,atol=1e-08)

		def lnlkl(x):
			diff = x - self.mean
			return -1./2.*diff.dot(self.precision).dot(diff)

		self.lnlkl = lnlkl

		return self

	@classmethod
	def GaussianRandom(cls,diag=0.2,ndim=2,seed=42):

		rng = random.RandomState(seed=seed)
		mean = rng.uniform(0.,1.,ndim)
		Q = rng.uniform(0.,1.,(ndim,ndim))
		covariance = Q.dot(Q) + diag*scipy.eye(ndim) # add eye to make the covariane matrix more diagonal

		return cls.Gaussian(mean=mean,covariance=covariance)

	@property
	def ndim(self):
		return len(self.argmax)

	@property
	def max_lnlkl(self):
		if not hasattr(self,'_max_lnlkl'):
			self._max_lnlkl = self.lnlkl(self.argmax)
		return self._max_lnlkl

	def delta_lnlkl(self,*args,**kwargs):
		return self.lnlkl(*args,**kwargs)-self.max_lnlkl

	def delta_chi2(self,*args,**kwargs):
		return -2.*self.delta_lnlkl(*args,**kwargs)

	def sample(self,nwalkers=None,sigma=None,nsteps=10000,progress=True,thin_by=10,**kwargs):
		if nwalkers is None: nwalkers = 10 * self.ndim
		sampler = emcee.EnsembleSampler(nwalkers,self.ndim,self.lnlkl)
		if sigma is None: sigma = scipy.diag(self.covariance)
		pst = emcee.utils.sample_ball(self.argmax,sigma,nwalkers)
		sampler.run_mcmc(pst,nsteps,progress=progress,thin_by=thin_by,**kwargs)
		return sampler


class BaseEstimator(object):

	logger = logging.getLogger('BaseEstimator')

	def __init__(self,likelihood,**kwargs):
		self.likelihood = likelihood
		self.__dict__.update(kwargs)

	def __call__(self):
		raise ValueError('You should start by implementing a __call__ method!')

	@property
	def ndim(self):
		return self.likelihood.ndim

	def delta_chi2(self,nsigmas):
		return stats.chi2.ppf(level(nsigmas),self.ndim)

	def _fit_gaussian_(self,xp,yp,pini=None):
		# to be improved...

		np = len(xp)
		self.logger.info('We use {} points for the fit.'.format(np))
		precision = scipy.zeros((self.ndim,self.ndim))

		def cost(p):
			ix = 0
			for iaxis1 in range(self.ndim):
				for iaxis2 in range(iaxis1,self.ndim):
					# Diagonal terms should be > 0
					if (iaxis1 == iaxis2) & (p[ix] < 0): return scipy.inf
					precision[iaxis1,iaxis2] = precision[iaxis2,iaxis1] = p[ix]
					ix += 1
			# Matrix should be positive-definite
			try:
				cho = linalg.cholesky(precision)
			except:
				return scipy.inf
			# Compute "theoretical" likelihood
			theory = scipy.zeros(np)
			data = scipy.zeros(np)
			argmax = self.likelihood.argmax
			for ix,(x,y) in enumerate(zip(xp,yp)):
				diff = x - argmax
				theory[ix] = diff.dot(precision).dot(diff)
				data[ix] = y
			# Return difference
			return scipy.sum((theory - data)**2)

		if pini is None:
			pini = []
			for iaxis1 in range(self.ndim):
				for iaxis2 in range(iaxis1,self.ndim):
					if iaxis1 == iaxis2:
						pini.append(1.)
					else:
						pini.append(0.)
		optimize.minimize(cost,pini,method='Nelder-Mead',options={'maxiter':100000})

		return precision

def classical_double_derivative_ij(fun,xp=(0.,0.),step=(0.1,0.1)):
	xp = scipy.array(xp)
	step = scipy.array(step)
	hi = step.copy(); hi[1] = 0.
	hj = step.copy(); hj[0] = 0.
	return (fun(*(xp+hi+hj)) - fun(*(xp-hi+hj)) - fun(*(xp+hi-hj)) + fun(*(xp-hi-hj)))/(4.*scipy.prod(step))

def classical_double_derivative_ii(fun,xp=0.,step=0.1):
	h = step
	return (fun(xp+2.*h) - 2.*fun(xp) + fun(xp-2.*h))/(4.*h**2)

class FisherEstimator(BaseEstimator):

	logger = logging.getLogger('FisherEstimator')

	def __init__(self,likelihood,**kwargs):
		super(FisherEstimator,self).__init__(likelihood=likelihood,**kwargs)
		self._xp,self._yp = [],[] # will contain all points where the likelihood is evaluated
	
	def __call__(self,step=0.1):
		argmax = self.likelihood.argmax
		fisher = scipy.zeros((self.ndim,self.ndim))
		for i in range(self.ndim):
			for j in range(i,self.ndim):
				vec = argmax.copy()
				if (i == j):
					def fun(x):
						vec[i] = x
						return -self.likelihood.lnlkl(vec)
					fisher[i,i] = classical_double_derivative_ii(fun,xp=argmax[i],step=step)
				else:
					def fun(x,y):
						vec[i] = x; vec[j] = y
						return -self.likelihood.lnlkl(vec)
					fisher[i,j] = classical_double_derivative_ij(fun,xp=(argmax[i],argmax[j]),step=(step,step))
					fisher[j,i] = fisher[i,j]
		return Likelihood.Gaussian(mean=self.likelihood.argmax,precision=fisher)

class LatinHypercubeEstimator(BaseEstimator):
	# Sample the parameter space with a latin hypercube sampling, and fit gaussians
	# on this lieklihood, or perform an interpolations between the points

	logger = logging.getLogger('LatinHypercubeEstimator')

	def __init__(self,likelihood,ranges,npts,**kwargs):
		self.likelihood = likelihood
		self.ranges = ranges  # array of shape (self.nim, 2), contains the boundaries of the cube to sample
		self.npts = npts  # number of points used to sample the cube
		self._xp, self._yp = [], [] # will contain all points sampled by the LHS

	def lhssample(self):
		self.logger.info('Generate one realization of the LHS.')
		self._xp = scipy.random.uniform(size=[self.ndim, self.npts])
		for idim in range(0, self.ndim):
			self._xp[idim] = (scipy.argsort(self._xp[idim])+0.5)/self.npts
		for ipt in range(0, self.npts):
			#self._xp[:, ipt] = (self._xp[:, ipt] - 1./2. + self.likelihood.argmax) * (self.ranges[1] - self.ranges[0])
			self._xp[:, ipt] = (self._xp[:, ipt]) * (self.ranges[1] - self.ranges[0]) + self.ranges[0]
		self._xp = self._xp.T

	def optim_lhssample(self):
		self.logger.info('Find optimum LHS.')
		ntest = 1000
		xx = scipy.zeros([self.ndim, self.ndim])
		max_minlength = 0
		for i in range(ntest):
			xx = scipy.random.uniform(size=[self.ndim, self.npts])
			for idim in range(0, self.ndim):
				xx[idim] = (scipy.argsort(xx[idim])+0.5)/self.npts
			lenght = min(pdist(xx.T))
			if lenght > max_minlength:
				max_minlength = lenght
				self._xp = xx.T
				break


	def __call__(self):

		# self.lhssample()
		self.optim_lhssample()
		self._yp = scipy.array(map(self.likelihood.delta_chi2,self._xp))
		self.logger.info('Fitting Gaussian...')
		precision = self._fit_gaussian_(self._xp,self._yp)
		return Likelihood.Gaussian(mean=self.likelihood.argmax,precision=precision)


class SliceEstimator(BaseEstimator):

	logger = logging.getLogger('SliceEstimator')

	def __init__(self,likelihood,**kwargs):
		super(SliceEstimator,self).__init__(likelihood=likelihood,**kwargs)
		self._xp,self._yp = [],[] # will contain all points where the likelihood is evaluated
		self.xp,self.yp = {},{} # will contain only the points of major interest

	def _delta_chi2_(self,x):
		y = self.likelihood.delta_chi2(x)
		self._xp.append(x)
		self._yp.append(y)
		return y

	def _find_xy_points_(self,nsigmas=2.):

		target = self.delta_chi2(nsigmas)
		for iaxis in range(self.ndim):
			vec = scipy.copy(self.likelihood.argmax)
			argmax = vec[iaxis]
			def cost(p):
				if p[0] < argmax: # We consider only the "right" part of the 1D axis
					return scipy.inf
				vec[iaxis] = p[0]
				return (self._delta_chi2_(vec) - target)**2
			optimize.minimize(cost,argmax,method='Nelder-Mead') #options={'maxiter':1000000,'fatol':1e-15}
			key = '_%s-%s' % (iaxis,nsigmas)
			self.xp['u' + key] = self._xp[-1]
			self.yp['u' + key] = self._yp[-1]
			opposite = scipy.copy(self._xp[-1]); opposite[iaxis] = 2*argmax-self._xp[-1][iaxis]
			self.xp['d' + key] = opposite
			self.yp['d' + key] = self._delta_chi2_(opposite)

	def _find_diag_points_(self,nsigmas=2.):

		target = self.delta_chi2(nsigmas)
		for iaxis1 in range(self.ndim):
			for iaxis2 in range(iaxis1+1,self.ndim):
				argmax = self.likelihood.argmax[iaxis1]
				a = ((self.xp['u_%s-%s' % (iaxis2, nsigmas)][iaxis2] - self.likelihood.argmax[iaxis2])
					/ (self.xp['u_%s-%s' % (iaxis1, nsigmas)][iaxis1] - self.likelihood.argmax[iaxis1]))
				vec = scipy.copy(self.likelihood.argmax)
				argmax1,argmax2 = vec[iaxis1],vec[iaxis2]
				def cost(p):
					if p[0] < argmax: # We consider only the "right" part of the 1D axis
						return scipy.inf
					vec[iaxis1] = p[0]
					vec[iaxis2] = (p[0] - argmax1) * a + argmax2
					return (self._delta_chi2_(vec) - target)**2
				optimize.minimize(cost,argmax,method='Nelder-Mead') #options={'maxiter':1000000,'fatol':1e-15}
				# Add point to list
				self.xp['u_%s-u_%s-%s' % (iaxis1,iaxis2,nsigmas)] = self._xp[-1]
				self.yp['u_%s-u_%s-%s' % (iaxis1,iaxis2,nsigmas)] = self._yp[-1]
				# Add also the symetrical point
				opposite = scipy.copy(self._xp[-1])
				opposite[iaxis1] = 2.*argmax1 - self._xp[-1][iaxis1]
				opposite[iaxis2] = 2.*argmax2 - self._xp[-1][iaxis2]
				self.xp['d_%s-d_%s-%s' % (iaxis1,iaxis2,nsigmas)] = opposite
				self.yp['d_%s-d_%s-%s' % (iaxis1,iaxis2,nsigmas)] = self._delta_chi2_(opposite)
				# Do all the same, but for the "anti-diagonal" i.e. opposite coefficient
				a *= -1
				optimize.minimize(cost,argmax,method='Nelder-Mead') #options={'maxiter':1000000,'fatol':1e-15}
				# Add point to list
				self.xp['u_%s-d_%s-%s' % (iaxis1,iaxis2,nsigmas)] = self._xp[-1]
				self.yp['u_%s-d_%s-%s' % (iaxis1,iaxis2,nsigmas)] = self._yp[-1]
				# Add also the symetrical point
				opposite = scipy.copy(self._xp[-1])
				opposite[iaxis1] = 2.*argmax1 - self._xp[-1][iaxis1]
				opposite[iaxis2] = 2.*argmax2 - self._xp[-1][iaxis2]
				self.xp['d_%s-u_%s-%s' % (iaxis1,iaxis2,nsigmas)] = opposite
				self.yp['d_%s-u_%s-%s' % (iaxis1,iaxis2,nsigmas)] = self._delta_chi2_(opposite)

	def __call__(self,use_all=False,nsigmas=[0.5,1.5,2.]):
		for nsigma in nsigmas:
			self.logger.info('Finding points for nsigmas = {:.2f}.'.format(nsigma))
			self._find_xy_points_(nsigmas=nsigma)
			self._find_diag_points_(nsigmas=nsigma)
		self.logger.info('Fitting Gaussian...')
		if use_all:
			xp = self._xp
			yp = self._yp
		else:
			keys = self.xp.keys()
			xp = [self.xp[key] for key in keys]
			yp = [self.yp[key] for key in keys]
		pini = []
		nsigmas = nsigmas[scipy.argmin(scipy.array(nsigmas)-1)]
		for iaxis1 in range(self.ndim):
			for iaxis2 in range(iaxis1,self.ndim):
				if iaxis1 == iaxis2:
					pini.append(1./(self.xp['u_%s-%s' % (iaxis1,nsigmas)][iaxis1] - self.likelihood.argmax[iaxis1])**2.)
				else:
					pini.append(0.)
		precision = self._fit_gaussian_(xp,yp,pini=pini)
		return Likelihood.Gaussian(mean=self.likelihood.argmax,precision=precision)


def plot_ellipses(sampler,discard=5000,gaussians=[],labels=[]):

	# Compare the deduced Fisher ellipse to the input one
	sigs_ellipse = scipy.array([1., 2.])
	al = scipy.sqrt(stats.chi2.ppf(special.erf(sigs_ellipse / scipy.sqrt(2.)), 2))
	t = scipy.linspace(0., 2. * constants.pi, 1000, endpoint=False)[:, None]
	ct = scipy.cos(t)
	st = scipy.sin(t)

	levels = map(level,sigs_ellipse)
	chain = sampler.get_chain(flat=True,discard=discard)
	fig = corner.corner(chain,levels=levels,plot_datapoints=False,plot_density=False,bins=100)
	colors = ['red', 'blue', 'green']
	for igauss,gaussian in enumerate(gaussians):
		axs = fig.axes
		covariance = gaussian.covariance
		ix = 0
		for iaxis1 in range(gaussian.ndim):
			for iaxis2 in range(gaussian.ndim):
				if iaxis1 == iaxis2:
					hist = scipy.histogram(chain[:, iaxis1], bins=100)
					x = scipy.linspace(hist[1][0],hist[1][-1],1001)
					y = scipy.exp(-(x-gaussian.mean[iaxis1])**2./2./covariance[iaxis1,iaxis1])
					axs[ix].plot(x,y*hist[0].max(),color=colors[igauss],label=labels[igauss],ls='--')
				elif iaxis1 > iaxis2:
					sigx2 = covariance[iaxis2,iaxis2]
					sigy2 = covariance[iaxis1,iaxis1]
					sigxy = covariance[iaxis2,iaxis1]
					a = al * scipy.sqrt(0.5 * (sigx2 + sigy2) + scipy.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
					b = al * scipy.sqrt(0.5 * (sigx2 + sigy2) - scipy.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
					th = 0.5 * scipy.arctan2(2. * sigxy, sigx2 - sigy2)
					x = gaussian.mean[iaxis2] + a * ct * scipy.cos(th) - b * st * scipy.sin(th)
					y = gaussian.mean[iaxis1] + a * ct * scipy.sin(th) + b * st * scipy.cos(th)
					axs[ix].plot(x,y,color=colors[igauss],ls='--')
				else:
					axs[ix].axis('off')
				ix += 1
	axs[0].legend(**{'loc':'upper left','ncol':1,'fontsize':16,'framealpha':0.5,'frameon':True,'bbox_to_anchor':(1.04,1.)})

def squeezed_lnlkl(self,x):
	diff = x - self.mean
	precision = self.precision.copy()
	factor = scipy.sqrt(1 + scipy.absolute(x[1] - self.mean[1])**1.)
	precision[0, :] *= factor
	precision[:, 0] *= factor
	return -1./2.*diff.dot(precision).dot(diff)


if __name__ == '__main__':

	logging.basicConfig(level=logging.INFO)

	likelihood = Likelihood.GaussianRandom(ndim=3,seed=42)

	# ndim = 3
	# mean = scipy.ones(ndim)
	# covariance = scipy.eye(ndim)
	# likelihood = Likelihood.Gaussian(mean=mean,covariance=covariance)
	likelihood.lnlkl = lambda x: squeezed_lnlkl(likelihood,x) # I give a name to my lambda function and I don't care


	npts = 100
	# /!\ The problem with LHS is that you don't now the range of interest of
	# the likelihood a priori
	# ranges = scipy.array([[0.5, 0.5, 0.5] , [1.5, 1.5, 1.5]])
	# ranges = scipy.array([[-1., -1., -1.] , [1., 1., 1.]])
	ranges = scipy.array([likelihood.argmax-0.5, likelihood.argmax+0.5])

	fisher_estimator = FisherEstimator(likelihood=likelihood)
	fisher_estimation = fisher_estimator(step=0.2)

	slice_estimator = SliceEstimator(likelihood=likelihood)
	slice_estimation = slice_estimator(nsigmas=[1.])

	lhs_estimator = LatinHypercubeEstimator(likelihood=likelihood,ranges=ranges,npts=npts)
	lhs_estimation = lhs_estimator()

	sampler = likelihood.sample(nsteps=int(5e3))
	plot_ellipses(sampler,gaussians=[fisher_estimation,slice_estimation,lhs_estimation],labels=['Fisher','Slice','LHS'],discard=500)
	#plot_ellipses(sampler,gaussians=[fisher_estimation],discard=500)
	# plot_ellipses(sampler,gaussian=slice_estimation,discard=500)
	pyplot.show()

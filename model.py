import math
import numpy as np

class COPAN(object):

    def __init__(self):
        self.params = {'Sigma': 1, 'Cstar': 1, 'm': 1.5, 'delta': 1, 'a0': 2.3,
                       'aT': 9.5, 'l0': 11.0, 'lT': 8.0, 'p': 4, 'Wp': 1, 'q0': 1,
                       'qT': 0, 'qP': 0, 'b': 0.36}
        # Parameters to vary
        self.params['y'] = 0.21667
        self.params['wL'] = 0

    def RHS(self, x):
        assert isinstance(x, np.ndarray)
        assert x.shape[-1]==2

        Pscale = 10 #P is on a smaller scale, so scale it up

        L = x[...,0]
        P = x[...,1]/Pscale

        A = (self.params['Cstar'] - L)/(1 + self.params['m'])

        Ldot = L*(((self.params['l0'] - (self.params['lT'] * A)) * np.sqrt(A/self.params['Sigma']))
                  - (self.params['a0'] + self.params['aT'] * A)) - (self.params['b'] * (L**(0.25)) * (P**(0.75)))


        rep_quant = (self.params['y']*(L**(0.25))) + (self.params['wL'] * L * (P**(0.25))/ self.params['Sigma'])

        Pdot = ((2*self.params['p']*self.params['Wp']* (P**(1.25)) * rep_quant) /
                   (((P**(0.5)) * (self.params['Wp']**2)) + (rep_quant**2))) - (self.params['q0'] * (P**(1.25))/ rep_quant)

        return np.array([Ldot.T, Pscale * Pdot.T]).T

    def RHS_Normalised(self, x):
        assert isinstance(x, np.ndarray)
        assert x.shape[-1]==2

        Pscale = 10 #P is on a smaller scale, so scale it up

        L = x[...,0]
        P = x[...,1]/Pscale

        A = (self.params['Cstar'] - L)/(1 + self.params['m'])

        Ldot = L*(((self.params['l0'] - (self.params['lT'] * A)) * np.sqrt(A/self.params['Sigma']))
                  - (self.params['a0'] + self.params['aT'] * A)) - (self.params['b'] * (L**(0.25)) * (P**(0.75)))


        rep_quant = (self.params['y']*(L**(0.25))) + (self.params['wL'] * L * (P**(0.25))/ self.params['Sigma'])

        Pdot = np.zeros(P.shape)
        Pdot[rep_quant!=0] = ((2*self.params['p']*self.params['Wp']* (P[rep_quant!=0]**(1.25)) * rep_quant[rep_quant!=0]) /
                   (((P[rep_quant!=0]**(0.5)) * (self.params['Wp']**2)) +
                    (rep_quant[rep_quant!=0]**2))) - (self.params['q0'] * (P[rep_quant!=0]**(1.25))/ rep_quant[rep_quant!=0])

        unnormalised = np.array([Ldot.T, Pscale * Pdot.T]).T

        normalisation = np.linalg.norm(unnormalised)**2

        return unnormalised/(1 + normalisation)
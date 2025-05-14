from numba import njit
import numpy as np
import os
from typing import Union, Optional, Generic

from few.utils.mappings.jacobian import ELdot_to_PEdot_Jacobian
from few.waveform.base import SphericalHarmonicWaveformBase
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq, AmpInterpSchwarzEcc
from few.waveform.waveform import KerrEccentricEquatorial
from few.trajectory.ode import KerrEccEqFlux
from few.trajectory.inspiral import EMRIInspiral

from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.fdinterp import FDInterpolatedModeSum
from few.utils.modeselector import ModeSelector, NeuralModeSelector

from few.utils.baseclasses import BackendLike

@njit
def dpdL(a, p, e, x):
    #ELdot_to_PEdot_Jacobian: Edot, Ldot -> pdot, edot.
    #pdot = dp/dE*Edot + dp/dL*Ldot
    #=> pdot = dp/dL when Edot = 0, Ldot = 1
    pdot, _ = ELdot_to_PEdot_Jacobian(a, p, e, x, 0, 1)
    return pdot

@njit
def dedL(a, p, e, x):
    #ELdot_to_PEdot_Jacobian: Edot, Ldot -> pdot, edot.
    #edot = de/dE*Edot + de/dL*Ldot
    #=> edot = de/dL when Edot = 0, Ldot = 1
    _, edot = ELdot_to_PEdot_Jacobian(a, p, e, x, 0, 1)
    return edot
    
class JointRelKerrEccFlux(KerrEccEqFlux):

    def add_fixed_parameters(self, m1: float, m2: float, a: float, additional_args=None):
        self.massratio = m1 * m2 / (m1 + m2) ** 2
        self.a = a
        
        self.Al, self.nl, self.Ag, self.ng = additional_args

        if additional_args is None:
            self.num_add_args = 0
        else:
            self.num_add_args = len(additional_args)

    def modify_rhs(
        self, ydot: np.ndarray, y: np.ndarray, **kwargs
    ) -> None:
        """
        This function allows the user to modify the right-hand side of the ODE after any required Jacobian transforms
        have been applied.

        By default, this function returns the input right-hand side unchanged.
        """

        #y = [p, e, x, pp, pt, pr]
        #ydot = [pdot, edot, xdot, Op, Ot, Or]

        p, e, x = y[:3]

        #calculate Ldot due to extra effects here
        Ldot_local = self.Al * (p / 10.) ** self.nl
        
        Ldot_global = self.Ag * (p) ** self.ng

        #convert Ldot to pdot, edot: pdot = dp/dL * Ldot, edot = de/dL * Ldot
        pdot_local = Ldot_local * dpdL(self.a, p, e, x)
        pdot_global = Ldot_global * dpdL(self.a, p, e, x)

        edot_local = Ldot_local * dedL(self.a, p, e, x)
        edot_global = Ldot_global * dedL(self.a, p, e, x)

        ydot[0] += pdot_local + pdot_global #in-place modification of pdot
        ydot[1] += edot_local + edot_global #in-place modification of pdot
        
class JointKerrWaveform(SphericalHarmonicWaveformBase, KerrEccentricEquatorial):

    def __init__(
        self,
        /,
        inspiral_kwargs: Optional[dict] = None,
        amplitude_kwargs: Optional[dict] = None,
        sum_kwargs: Optional[dict] = None,
        Ylm_kwargs: Optional[dict] = None,
        mode_selector_kwargs: Optional[dict] = None,
        force_backend: BackendLike = None,
        **kwargs: dict,
        ):
        if inspiral_kwargs is None:
            inspiral_kwargs = {}
        inspiral_kwargs["func"] = JointRelKerrEccFlux #modified Inspiral Class

        if sum_kwargs is None:
            sum_kwargs = {}
        mode_summation_module = InterpolatedModeSum
        if "output_type" in sum_kwargs:
            if sum_kwargs["output_type"] == "fd":
                mode_summation_module = FDInterpolatedModeSum

        if mode_selector_kwargs is None:
            mode_selector_kwargs = {}
        mode_selection_module = ModeSelector
        if "mode_selection_type" in mode_selector_kwargs:
            if mode_selector_kwargs["mode_selection_type"] == "neural":
                mode_selection_module = NeuralModeSelector
                if "mode_selector_location" not in mode_selector_kwargs:
                    mode_selector_kwargs["mode_selector_location"] = os.path.join(
                        dir_path,
                        "./files/modeselector_files/KerrEccentricEquatorialFlux/",
                    )
                mode_selector_kwargs["keep_inds"] = np.array(
                    [0, 1, 2, 3, 4, 6, 7, 8, 9]
                )

        KerrEccentricEquatorial.__init__(
            self,
            **{
                key: value
                for key, value in kwargs.items()
                if key in ["lmax", "nmax", "ndim"]
            },
            force_backend=force_backend,
        )
        SphericalHarmonicWaveformBase.__init__(
            self,
            inspiral_module=EMRIInspiral,
            amplitude_module=AmpInterpKerrEccEq,
            sum_module=mode_summation_module,
            mode_selector_module=mode_selection_module,
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            sum_kwargs=sum_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            mode_selector_kwargs=mode_selector_kwargs,
            **{
                key: value for key, value in kwargs.items() if key in ["normalize_amps"]
            },
            force_backend=force_backend,
        )

    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    @property
    def allow_batching(self):
        return False

    def __call__(
        self,
        M: float,
        mu: float,
        a: float,
        p0: float,
        e0: float,
        xI: float,
        theta: float,
        phi: float,
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ) -> np.ndarray:
        """
        Generate the waveform.

        Args:
            M: Mass of larger black hole in solar masses.
            mu: Mass of compact object in solar masses.
            a: Dimensionless spin of massive black hole.
            p0: Initial semilatus rectum of inspiral trajectory.
            e0: Initial eccentricity of inspiral trajectory.
            xI: Initial cosine of the inclination angle.
            theta: Polar angle of observer.
            phi: Azimuthal angle of observer.
            *args: Placeholder for additional arguments.
            **kwargs: Placeholder for additional keyword arguments.

        Returns:
            Complex array containing generated waveform.

        """
        return self._generate_waveform(
            M,
            mu,
            a,
            p0,
            e0,
            xI,
            theta,
            phi,
            *args,
            **kwargs,
        )

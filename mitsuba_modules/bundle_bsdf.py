from typing import Tuple
import mitsuba as mi
import drjit as dr
mi.set_variant("llvm_ad_rgb")

from mitsuba import BSDF, BSDFContext, BSDFSample3f, Color3f, Point2f, SurfaceInteraction3f, Vector3f
import numpy as np


class BundleBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.model, self.passthrough_chance = load_model("./model")
        self.model_tensor = mi.TensorXf(self.model)
        
        # Read 'eta' and 'tint' properties from `props`
        self.eta = 1.33
        if props.has_property('eta'):
            self.eta = props['eta']

        self.tint = props['tint']

        # Set the BSDF flags
        reflection_flags   = mi.BSDFFlags.DeltaReflection   | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components  = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags

    def sample(self, ctx, si, sample1, sample2, active):
        # Compute Fresnel terms
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        r_i, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, self.eta)
        t_i = dr.maximum(1.0 - r_i, 0.0)

        # Pick between reflection and transmission
        selected_r = (sample1 <= r_i) & active

        # Pick between reflection and passthrough
        is_interacted = (sample1 >= self.passthrough_chance) & active

        # Fill up the BSDFSample struct
        bs = mi.BSDFSample3f()
        bs.pdf = dr.select(selected_r, r_i, t_i)
        bs.sampled_component = dr.select(is_interacted, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type      = dr.select(is_interacted, mi.UInt32(+mi.BSDFFlags.DeltaTransmission),
                                                     mi.UInt32(+mi.BSDFFlags.Null))
        bs.wo = dr.select(is_interacted,
                          mi.reflect(si.wi),
                          si.wi)
        # bs.eta = dr.select(selected_r, 1.0, eta_it)
        # TODO: is this correct?
        bs.eta = 1.0

        # For reflection, tint based on the incident angle (more tint at grazing angle)
        value_r = dr.lerp(mi.Color3f(self.tint), mi.Color3f(1.0), dr.clamp(cos_theta_i, 0.0, 1.0))

        # For transmission, radiance must be scaled to account for the solid angle compression
        value_t = mi.Color3f(1.0) # * dr.sqr(eta_ti)

        value = dr.select(selected_r, value_r, value_t)

        return (bs, value)

    def eval(self, ctx, si, wo, active):
        return self.pdf(ctx,si,wo,active)

    def pdf(self, ctx, si, wo, active):
        # TODO: Get the directions from si and wo
        return self.eval_model(ctx,si,wo,active)

    def eval_pdf(self, ctx, si, wo, active):
        return self.eval(ctx,si,wo,active), self.pdf(ctx,si,wo,active)
    
    def eval_pdf_sample(self: BSDF, ctx: BSDFContext, si: SurfaceInteraction3f, wo: Vector3f, sample1: float, sample2: Point2f, active: bool = True) -> Tuple[Color3f, float, BSDFSample3f, Color3f]:
        return super().eval_pdf_sample(ctx, si, wo, sample1, sample2, active)

    def traverse(self, callback):
        callback.put_parameter('tint', self.tint, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        return ('BundleBSDF[\n'
                '    passthrough chance=%s,\n'
                ']' % (self.passthrough_chance))
    
    def eval_model(self, theta_in: mi.Float, phi_in: mi.Float, theta_out: mi.Float, phi_out: mi.Float, wavelength: mi.Float, active) -> mi.Float:

        t_i_coord, phi_i_coord, t_o_coord, phi_o_coord, wave_coord = self.get_model_coords(theta_in, phi_in, theta_out, phi_out, wavelength)
        
        index = self.coords_to_single_dim(t_i_coord, phi_i_coord, t_o_coord, phi_o_coord, wave_coord)
        
        return dr.gather(mi.Float, self.model_tensor.array, index, active)
    
    def get_model_coords(self, theta_in: mi.Float, phi_in: mi.Float, theta_out: mi.Float, phi_out: mi.Float, wavelength: mi.Float) -> tuple[mi.UInt32, mi.UInt32, mi.UInt32, mi.UInt32, mi.UInt32]:
        shape_theta_i = self.model.shape[0]
        shape_phi_i = self.model.shape[1]
        shape_theta_o = self.model.shape[2]
        shape_phi_o = self.model.shape[3]
        shape_wave = self.model.shape[4]

        # TODO: Turn float into uint indices
        return 

    def coords_to_single_dim(self, theta_i: mi.UInt32, phi_i: mi.UInt32, theta_o: mi.UInt32, phi_o: mi.UInt32, wavelength: mi.UInt32) -> mi.UInt32:
        # TODO: Turn 5 uint indices into 1 uint index
        # use shapes in reverse order?
        return mi.UInt32(0)
    
def load_model(filepath) -> tuple[np.array, float]:
    # Load the model from the files in the following order
    # Theta_in: [0,180] steps of 10
    # Phi_in: [0,0]
    # Theta_out: [0,180] 200 steps
    # Phi_out: [0,360] 200 steps
    # Wavelength: [0,24] steps of 1

    # Returns the model as a numpy array, as well as a float in [0,1], which is the chance of interaction
    model = np.zeros((18,1,200,200,25))

    # go through all the folders called theta-y-phi-0
    folders = [f'theta-{x}-phi-0' for x in range(0,180,10)]
    for (j,folder) in enumerate(folders):
        files = [f'lambda_{y}_intensities.npy' for y in range(25)]
        for (i,file) in enumerate(files):
            data = np.load(f'./mitsuba_modules/model/{folder}/{file}')
            data = data.reshape((200,200))
            model[j,0,:,:,i] = data
    # Read chance from chance.txt
    chance_file = open("./mitsuba_modules/model/chance.txt")
    chance = float(chance_file.read().strip())

    return model, chance
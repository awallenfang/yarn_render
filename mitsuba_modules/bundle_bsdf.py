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
        # self.model, self.passthrough_chance = load_single_model("./single_model")
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
        self.m_flags = reflection_flags | transmission_flags | mi.BSDFFlags.Smooth

    def sample(self, ctx, si, sample1, sample2, active):
        # Pick between reflection and passthrough
        passthrough = (sample1 <= self.passthrough_chance) & active
        interacted_dir = mi.warp.square_to_uniform_sphere(sample2)

        # Fill up the BSDFSample struct
        bs = mi.BSDFSample3f()
        bs.pdf = dr.select(passthrough,  self.passthrough_chance, self.pdf(ctx, si, interacted_dir, active))
        bs.sampled_component = dr.select(passthrough,  mi.UInt32(1), mi.UInt32(0),)
        bs.sampled_type      = dr.select(passthrough, mi.UInt32(+mi.BSDFFlags.Null) + mi.UInt32(+mi.BSDFFlags.Delta), mi.UInt32(+mi.BSDFFlags.DeltaTransmission) + mi.UInt32(+mi.BSDFFlags.Smooth))
        bs.wo = dr.select(passthrough,
                          -si.wi,
                          interacted_dir)

        # TODO: is this correct?
        bs.eta = 1.0

        local_in = si.to_local(dr.normalize(si.wi))
        theta_in = dr.asin(local_in.z)
        phi_in = dr.atan2(local_in.y, local_in.x)
        phi_in[phi_in < 0] += 2*dr.pi
        # dr.printf_async("theta_in: %s, phi_in: %s\n", theta_in, phi_in, active=active)
        # value = dr.select(passthrough, 1. * self.passthrough_chance, si.uv.y / bs.pdf )
        # value = dr.select(passthrough, 1. * self.passthrough_chance, self.eval(ctx, si, interacted_dir, active))
        value = dr.select(passthrough, 1., self.eval(ctx, si, interacted_dir, active) / bs.pdf)
        # value = dr.select(passthrough, 1. * self.passthrough_chance, phi_in / (2. * dr.pi))

        # value = dr.select(passthrough, 1. * self.passthrough_chance, theta_in / (dr.pi))

        return (bs, value)

    def eval(self, ctx, si, wo, active):
        incident_dir = dr.normalize(-si.wi)
        wavelengths = si.wavelengths
        outgoing_dir = dr.normalize(wo)

        # theta_in = dr.acos(si.to_local(incident_dir).z) + dr.pi/2
        theta_in = dr.asin(si.to_local(incident_dir).z) + dr.pi/2
        phi_in = dr.atan2(si.to_local(incident_dir).y, si.to_local(incident_dir).x)
        phi_in[phi_in < 0] += 2*dr.pi

        # phi_in = si.uv.x * 2*dr.pi
        # theta_out = dr.acos(si.to_local(outgoing_dir).z) + dr.pi/2
        theta_out = dr.asin(si.to_local(outgoing_dir).z) + dr.pi/2
        phi_out = dr.atan2(si.to_local(outgoing_dir).y, si.to_local(outgoing_dir).x)
        phi_out[phi_out < 0] += 2*dr.pi

        # phi_out = dr.atan2(si.to_local(outgoing_dir).z, si.to_local(outgoing_dir).x)

        # Calculate the offset in this case, since the input model is circular
        # Also apply twisting
        phi_out_offset = (phi_out - phi_in) + dr.pi #+ si.uv.y * 20*dr.pi
        phi_out_offset[phi_out_offset < 0] += 2*dr.pi
        phi_out_offset[phi_out_offset > 2*dr.pi] -= 2*dr.pi

        # TODO: Check how wavelengths are handled 
        # return mi.Float(0.0001)
        # return (1. - self.passthrough_chance) * self.eval_model(theta_in, 0., theta_out, phi_out_offset, mi.Float(600.), active) * dr.dot(si.wi, si.n)
        return (1. - self.passthrough_chance) * self.eval_model(theta_in, 0., theta_out, phi_out_offset, mi.Float(600.), active)

    def pdf(self, ctx, si, wo, active):
        return (1. - self.passthrough_chance)  / (4. * dr.pi)
        
        # return self.eval_model(ctx,si,wo,active, 600., active)
        # return mi.Float(0.)

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
        t_i_coord_lower, t_i_coord_upper, factor , phi_i_coord, t_o_coord, phi_o_coord, wave_coord = self.get_model_coords(theta_in, phi_in, theta_out, phi_out, wavelength)
        
        index_l = self.coords_to_single_dim(t_i_coord_lower, phi_i_coord, t_o_coord, phi_o_coord, wave_coord)
        index_u = self.coords_to_single_dim(t_i_coord_upper, phi_i_coord, t_o_coord, phi_o_coord, wave_coord)
        
        value_l = dr.gather(mi.Float, self.model_tensor.array, index_l, active)
        value_u = dr.gather(mi.Float, self.model_tensor.array, index_u, active)

        return dr.lerp(value_l, value_u, factor)
    
    def get_model_coords(self, theta_in: mi.Float, phi_in: mi.Float, theta_out: mi.Float, phi_out: mi.Float, wavelength: mi.Float) -> tuple[mi.UInt32, mi.UInt32, mi.UInt32, mi.UInt32, mi.UInt32]:
        shape_theta_i = mi.UInt32(self.model.shape[0])
        shape_phi_i = mi.UInt32(self.model.shape[1])
        shape_theta_o = mi.UInt32(self.model.shape[2])
        shape_phi_o = mi.UInt32(self.model.shape[3])
        shape_wave = mi.UInt32(self.model.shape[4])

        # theta_i_coord = mi.UInt32((theta_in + dr.pi/2) / dr.pi * shape_theta_i)
        theta_i_coord_lower = mi.UInt32(dr.floor((theta_in ) / dr.pi * shape_theta_i))
        theta_i_coord_upper = mi.UInt32(dr.ceil((theta_in ) / dr.pi * shape_theta_i))
        phi_i_coord = mi.UInt32((phi_in) / (2*dr.pi) * shape_phi_i)
        # theta_o_coord = mi.UInt32((theta_out + dr.pi/2) / dr.pi * shape_theta_o)
        theta_o_coord = mi.UInt32((theta_out ) / dr.pi * shape_theta_o)
        phi_o_coord = mi.UInt32((phi_out) / (2*dr.pi) * shape_phi_o)
        wave_coord = mi.UInt32((wavelength - 400.) / 300. * shape_wave)

        # dr.printf_async("theta_i_coord: %s\n", theta_i_coord)
        factor = (theta_in / dr.pi) * shape_theta_i - mi.Float(theta_i_coord_lower)
        
        return theta_i_coord_lower, theta_i_coord_upper, factor, phi_i_coord, theta_o_coord, phi_o_coord, wave_coord

    def coords_to_single_dim(self, theta_i: mi.UInt32, phi_i: mi.UInt32, theta_o: mi.UInt32, phi_o: mi.UInt32, wavelength: mi.UInt32) -> mi.UInt32:
        # use shapes in reverse order?
        # Shape speed: t_i>p_i>t_o>p_o>wave

        # shape_theta_i = self.model.shape[0]
        shape_phi_i = mi.UInt32(self.model.shape[1])
        shape_theta_o = mi.UInt32(self.model.shape[2])
        shape_phi_o = mi.UInt32(self.model.shape[3])
        shape_wave = mi.UInt32(self.model.shape[4])

        return theta_i * shape_phi_i * shape_theta_o * shape_phi_o * shape_wave + \
                phi_i * shape_theta_o * shape_phi_o * shape_wave + \
                theta_o * shape_phi_o * shape_wave + \
                phi_o * shape_wave + \
                wavelength
    

### TESTS

def load_single_model(filepath) -> np.array:
    model = np.zeros((13,1,450,880,25))

    folders = [f'theta-{x}-phi-0' for x in range(0,190,15)]
    for (j,folder) in enumerate(folders):
        files = [f'fiber_0_lambda{y}_TM_depth6.binary' for y in range(25)]
        for (i,file) in enumerate(files):
            data = np.fromfile(f'./mitsuba_modules/single_model/{folder}/{file}', dtype="float32")
            data = data.reshape(450,880)
            model[j,0,:,:,i] = data

    # Normalize model for each wavelength seperately
    for i in range(25):
        model[:,:,:,:,i] /= np.max(model[:,:,:,:,i])

    return model, 0.00001


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
            # data = data.reshape((200,200))
            model[j,0,:,:,i] = data

    # Normalize model for each wavelength seperately
    for i in range(25):
        # print(np.sum(model[:,:,:,:,i]))
        model[:,:,:,:,i] /= np.max(model[:,:,:,:,i])
        # print(np.sum(model[:,:,:,:,i]))
    # Read chance from chance.txt
    chance_file = open("./mitsuba_modules/model/chance.txt")
    chance = float(chance_file.read().strip())

    return model, chance



def eval_model(model, model_tensor, theta_in: mi.Float, phi_in: mi.Float, theta_out: mi.Float, phi_out: mi.Float, wavelength: mi.Float, active) -> mi.Float:
    t_i_coord, phi_i_coord, t_o_coord, phi_o_coord, wave_coord = get_model_coords(model, theta_in, phi_in, theta_out, phi_out, wavelength)
    
    index = coords_to_single_dim(model, t_i_coord, phi_i_coord, t_o_coord, phi_o_coord, wave_coord)
    
    return dr.gather(mi.Float, model_tensor.array, index, active)

def get_model_coords(model, theta_in: mi.Float, phi_in: mi.Float, theta_out: mi.Float, phi_out: mi.Float, wavelength: mi.Float) -> tuple[mi.UInt32, mi.UInt32, mi.UInt32, mi.UInt32, mi.UInt32]:
    shape_theta_i = mi.UInt32(model.shape[0])
    shape_phi_i = mi.UInt32(model.shape[1])
    shape_theta_o = mi.UInt32(model.shape[2])
    shape_phi_o = mi.UInt32(model.shape[3])
    shape_wave = mi.UInt32(model.shape[4])

    theta_i_coord = mi.UInt32((theta_in + dr.pi/2) / dr.pi * shape_theta_i)
    phi_i_coord = mi.UInt32((phi_in) / (2*dr.pi) * shape_phi_i)
    # print((phi_in) / (2*dr.pi))
    # print(shape_phi_i)
    # print(phi_i_coord)
    theta_o_coord = mi.UInt32((theta_out + dr.pi/2) / dr.pi * shape_theta_o)
    phi_o_coord = mi.UInt32((phi_out) / (2*dr.pi) * shape_phi_o)
    wave_coord = mi.UInt32((wavelength - 400.) / 300. * shape_wave)

    # dr.printf_async("theta_i_coord: %s\n", theta_i_coord)
    
    return theta_i_coord, phi_i_coord, theta_o_coord, phi_o_coord, wave_coord

def coords_to_single_dim(model, theta_i: mi.UInt32, phi_i: mi.UInt32, theta_o: mi.UInt32, phi_o: mi.UInt32, wavelength: mi.UInt32) -> mi.UInt32:
    # use shapes in reverse order?
    # Shape speed: t_i>p_i>t_o>p_o>wave

    # shape_theta_i = self.model.shape[0]
    shape_phi_i = mi.UInt32(model.shape[1])
    shape_theta_o = mi.UInt32(model.shape[2])
    shape_phi_o = mi.UInt32(model.shape[3])
    shape_wave = mi.UInt32(model.shape[4])

    return theta_i * shape_phi_i * shape_theta_o * shape_phi_o * shape_wave + \
            phi_i * shape_theta_o * shape_phi_o * shape_wave + \
            theta_o * shape_phi_o * shape_wave + \
            phi_o * shape_wave + \
            wavelength

if __name__ == "__main__":

    model = load_model("./model")[0]
    tensor = mi.TensorXf(model)

    test = np.zeros((100,100))

    # Go through theta_out and phi_out and fill in the test array
    for t in range(100):
        print(t)
        for p in range(100):
            # print(t,p)
            test[t,p] = eval_model(model, tensor, (-np.pi / 4.), 0., (-np.pi / 2.) + t * (np.pi / 100.), p * ((2.*np.pi )/ 100.), mi.Float(500.), True)[0]

    print(eval_model(model, tensor, 0., 0., 0., 0., mi.Float(600.), True))

    import matplotlib.pyplot as plt
    plt.imshow(test)
    plt.show()
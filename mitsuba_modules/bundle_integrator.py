import mitsuba as mi
import drjit as dr

# Use flags analoguous to roughplastic with direct transmission and reflection/glossy

class BundleIntegrator(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        self.m_max_depth = props.get("max_depth", 100)

        self.m_shading_samples = props.get("shading_samples", 1)
        self.m_emitter_samples = props.get("emitter_samples", self.m_shading_samples) 
        self.m_bsdf_samples = props.get("bsdf_samples", self.m_shading_samples)

        sum = self.m_emitter_samples + self.m_bsdf_samples
        self.m_weight_bsdf = 1. / self.m_bsdf_samples
        self.m_weight_lum = 1. / self.m_emitter_samples
        self.m_frac_bsdf = self.m_bsdf_samples / sum
        self.m_frac_lum = self.m_emitter_samples / sum

    def sample(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, medium: mi.Medium, active: mi.Mask):
        
        ray = mi.Ray3f(ray)
        throughput: mi.Spectrum = mi.Spectrum(1.)
        result: mi.Spectrum = mi.Spectrum(0.)
        eta: mi.Float = mi.Float(1.)
        depth: mi.UInt32 = mi.UInt32(0)
        
        valid_ray: mi.Bool = mi.Bool(dr.neq(scene.environment(), None))

        prev_si: mi.Interaction3f = dr.zeros(mi.Interaction3f)
        prev_bsdf_pdf: mi.Float = mi.Float(1.)
        prev_bsdf_delta: mi.Bool = mi.Mask(True)
        bsdf_ctx: mi.BSDFContext = mi.BSDFContext()
        si = dr.zeros(mi.Interaction3f)
        ds = dr.zeros(mi.DirectionSample3f)
        em_pdf: mi.Float = mi.Float(0.)
        mis_bsdf: mi.Float = mi.Float(0.)
        active_next: mi.Bool = mi.Bool(True)
        active_em_sample: mi.Bool = mi.Bool(True)

        loop = mi.Loop("Bundle Path Tracer", lambda: (sampler, ray, throughput, result, si, ds, em_pdf, mis_bsdf,                           eta, depth, valid_ray, prev_si, prev_bsdf_pdf,
                            prev_bsdf_delta, active, active_next, active_em_sample))
        
        loop.set_max_iterations(self.m_max_depth)

        while loop(active):
            si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, dr.eq(depth, 0))


            # Sample emitters directly
            active_em_sample = dr.neq(si.emitter(scene), None)
            ds: mi.DirectionSample3f = mi.DirectionSample3f(scene, si, prev_si)
            em_pdf: mi.Float = mi.Float(0.)

            em_pdf[~prev_bsdf_delta ] = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            dr.printf_async("em_pdf: %f\n", em_pdf)
            dr.printf_async("prev: %f\n", prev_bsdf_pdf)

            mis_bsdf: mi.Float = self.mis_weight(prev_bsdf_pdf, em_pdf)

            result[active_em_sample] = dr.fma(throughput, ds.emitter.eval(si,prev_bsdf_pdf > 0.) * mis_bsdf, result)

            # Should we even continue tracing?
            active_next: mi.Bool = (depth +1 < self.m_max_depth) and si.is_valid()

            # if dr.none(active_next):
            #     break

            bsdf = si.bsdf(ray)

            ##### Emitter sampling #####

            active_em: mi.Mask = active_next and mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            ds: mi.DirectionSample3f = dr.zeros(mi.DirectionSample3f)
            em_weight: mi.Spectrum = dr.zeros(mi.Spectrum)
            wo: mi.Vector3f = dr.zeros(mi.Vector3f)

            ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)

            active_em &= dr.neq(ds.pdf, 0.)

            wo[active_em] = si.to_local(ds.d)

            ###### BSDF * cos(theta)

            sample_1: mi.Float = sampler.next_1d()
            sample_2: mi.Point2f = sampler.next_2d()

            bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2)

            #### Emitter sampling contribution ####
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

            mis_em = dr.select(ds.delta, 1., self.mis_weight(ds.pdf, bsdf_pdf))

            result[active_em] = dr.fma(ds.delta, bsdf_val * em_weight * mis_em, result)

            ###### BSDF Sampling #####
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

            # NOTE: Change position of this ray
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            
            ##### Update loop vars ######
                
            throughput *= bsdf_weight
            eta *= bsdf_sample.eta
            valid_ray |= active and si.is_valid() and not mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)

            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            #### Stopping Criterion #####
            depth[si.is_valid()] += 1

            throughput_max = dr.max(mi.unpolarized_spectrum(throughput))

            rr_prob = dr.min(throughput_max * dr.sqr(eta), 0.95)
            rr_active = depth >= self.m_rr_depth
            rr_continue = sampler.next_1d() < rr_prob

            throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))

            active = active_next and (not rr_active or rr_continue) and dr.neq(throughput_max, 0.)

            # Result, mask if ray exited scenes
            return (result, True)


    def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
        pdf_a *= pdf_a
        pdf_b *= pdf_b
        w = pdf_a / (pdf_a + pdf_b)
        return dr.select(dr.isfinite(w), w, 0.)
    
    def spec_fma(a: mi.Spectrum, b: mi.Spectrum, c: mi.Spectrum):
        return dr.fma(a,b,c)
    
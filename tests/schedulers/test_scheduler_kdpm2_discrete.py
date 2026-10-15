import torch

from diffusers import KDPM2DiscreteScheduler

from ..testing_utils import torch_device
from .test_schedulers import SchedulerCommonTest


class KDPM2DiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (KDPM2DiscreteScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_sigma_interpolation_zero_endpoints_stay_finite(self):
        # The schedule has a terminal sigma of zero. log(0).lerp(log(sigma))
        # can yield NaN even before the solver performs its final update.
        for schedule in (
            {},
            {"use_karras_sigmas": True},
            {"use_exponential_sigmas": True},
            {"use_beta_sigmas": True},
        ):
            scheduler = KDPM2DiscreteScheduler(**self.get_scheduler_config(**schedule))
            scheduler.set_timesteps(30)
            assert torch.isfinite(scheduler.sigmas_interpol).all()
            assert scheduler.sigmas_interpol[0] == 0
            assert scheduler.sigmas_interpol[-1] == 0
            assert torch.isfinite(scheduler.timesteps).all()

    def test_sigma_interpolation_reproduces_geometric_mean(self):
        # The stable formulation must still be the geometric midpoint
        # for strictly positive sigmas, including widely varying magnitudes.
        sigmas = torch.tensor([80.0, 10.0, 1e-12, 0.0], dtype=torch.float32)
        actual = sigmas.sqrt() * sigmas.roll(1).sqrt()
        expected = torch.tensor(
            [0.0, (80.0 * 10.0) ** 0.5, (10.0 * 1e-12) ** 0.5, 0.0],
            dtype=torch.float32,
        )
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-12)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if torch_device in ["cpu", "mps"]:
            assert abs(result_sum.item() - 4.6934e-07) < 1e-2
            assert abs(result_mean.item() - 6.1112e-10) < 1e-3
        else:
            # CUDA
            assert abs(result_sum.item() - 4.693428650170972e-07) < 1e-2
            assert abs(result_mean.item() - 0.0002) < 1e-3

    def test_full_loop_no_noise(self):
        if torch_device == "mps":
            return
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if torch_device in ["cpu", "mps"]:
            assert abs(result_sum.item() - 20.4125) < 1e-2
            assert abs(result_mean.item() - 0.0266) < 1e-3
        else:
            # CUDA
            assert abs(result_sum.item() - 20.4125) < 1e-2
            assert abs(result_mean.item() - 0.0266) < 1e-3

    def test_full_loop_device(self):
        if torch_device == "mps":
            return
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps, device=torch_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter.to(torch_device) * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if str(torch_device).startswith("cpu"):
            # The following sum varies between 148 and 156 on mps. Why?
            assert abs(result_sum.item() - 20.4125) < 1e-2
            assert abs(result_mean.item() - 0.0266) < 1e-3
        else:
            # CUDA
            assert abs(result_sum.item() - 20.4125) < 1e-2
            assert abs(result_mean.item() - 0.0266) < 1e-3

    def test_full_loop_with_noise(self):
        if torch_device == "mps":
            return
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        # add noise
        t_start = self.num_inference_steps - 2
        noise = self.dummy_noise_deter
        noise = noise.to(sample.device)
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        sample = scheduler.add_noise(sample, noise, timesteps[:1])

        for i, t in enumerate(timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 70408.4062) < 1e-2, f" expected result sum 70408.4062, but get {result_sum}"
        assert abs(result_mean.item() - 91.6776) < 1e-3, f" expected result mean 91.6776, but get {result_mean}"

    def test_beta_sigmas(self):
        self.check_over_configs(use_beta_sigmas=True)

    def test_exponential_sigmas(self):
        self.check_over_configs(use_exponential_sigmas=True)

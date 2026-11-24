"""One causal shared-weight rollout for all future object-state horizons."""
import torch
from torch import nn
import torch.nn.functional as F

from .instance_motion_model import decoded_boxes, forecast_loss

OUTPUT_HORIZONS = (0.5, 1., 2., 3., 5.)
OUTPUT_STEPS = (0, 1, 3, 5, 9)


def physical_displacements(features, bases, omega, seconds):
    """History-only motion experts at absolute time; never read future labels."""
    velocity = features[:, -5:-3] * 15
    acceleration = features[:, -3:-1] * 5
    angle = omega * seconds
    tangent = torch.stack((-velocity[:, 1], velocity[:, 0]), -1)
    turn = seconds * (torch.sinc(angle / torch.pi)[:, None] * velocity
                      + (.5 * angle * torch.sinc(angle / (2 * torch.pi)).square())[:, None] * tangent)
    direction = velocity / velocity.norm(dim=-1, keepdim=True).clamp_min(.1)
    longitudinal = (acceleration * direction).sum(-1, keepdim=True) * direction
    return torch.cat((bases[:, :3] * seconds,
                      (velocity * seconds + .5 * acceleration * seconds**2)[:, None],
                      (turn + .5 * longitudinal * seconds**2)[:, None]), 1)


class RecurrentInstanceMotionPredictor(nn.Module):
    """Free-running 0.5s transitions. Later predictions consume earlier states.

    States stay in each actor's CURRENT body frame, not a moving ego frame.
    Geometry association and input feature semantics match V8. No teacher forcing.
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(43, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.transition = nn.GRUCell(52, hidden)
        self.head = nn.Linear(hidden, 11)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = 2.
            self.head.bias[10] = 4.6

    def forward(self, features, bases, omega, steps=10):
        if not 1 <= steps <= 10:
            raise ValueError('supported rollout: 1..10 half-second steps')
        hidden = self.encoder(features)
        displacement = torch.zeros_like(bases[:, 0])
        yaw = torch.zeros_like(omega)
        log_size = torch.zeros_like(displacement)
        existence = torch.full_like(omega, 4.6)
        previous_experts = torch.zeros_like(bases)
        output = []
        for step in range(1, steps + 1):
            time = .5 * step
            timing = features.new_tensor([time / 5, .5]).expand(len(features), -1)
            state = torch.cat((features, displacement / 50, yaw.sin()[:, None], yaw.cos()[:, None],
                               log_size, existence.sigmoid()[:, None], timing), -1)
            hidden = self.transition(state, hidden)
            value = self.head(hidden)
            mixture = value[:, :5].softmax(-1)
            experts = physical_displacements(features, bases, omega, time)
            displacement = displacement + (mixture[:, :, None] * (experts - previous_experts)).sum(1) + .5 * value[:, 5:7]
            yaw = yaw + .5 * ((mixture[:, 3] + mixture[:, 4]) * omega + .8 * value[:, 7].tanh())
            log_size = .25 * value[:, 8:10].tanh()
            existence = value[:, 10]
            output.append((displacement, yaw, log_size, existence))
            previous_experts = experts
        return tuple(torch.stack([entry[k] for entry in output], 1) for k in range(4))


def rollout_boxes(current, predictions):
    displacement, yaw, log_size, _ = predictions
    n, steps = displacement.shape[:2]
    result = decoded_boxes(current[:, None].expand(-1, steps, -1).reshape(-1, 5),
                           displacement.reshape(-1, 2), yaw.reshape(-1), log_size.reshape(-1, 2))
    return result.reshape(n, steps, 5)


def multihorizon_loss(predictions, current, targets, valid):
    """Equal weight per requested horizon; each step normalizes over its valid actors."""
    losses = []
    for step in OUTPUT_STEPS:
        loss, _ = forecast_loss(tuple(value[:, step] for value in predictions), current,
                                targets[:, step], valid[:, step])
        losses.append(loss)
    return torch.stack(losses).mean(), torch.stack(losses).detach()

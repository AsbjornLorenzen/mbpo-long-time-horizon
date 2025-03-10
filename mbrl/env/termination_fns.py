# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch

# TODO remove act from all of these, it's not needed



def hopper(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        torch.isfinite(next_obs).all(-1)
        * (next_obs[:, 1:].abs() < 100).all(-1)
        * (height > 0.7)
        * (angle.abs() < 0.2)
    )

    done = ~not_done
    done = done[:, None]
    return done


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    not_done = torch.isfinite(next_obs).all(-1) * (next_obs[:, 1].abs() <= 0.2)
    done = ~not_done

    done = done[:, None]

    return done



def double_inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2
    
    def compute_second_pole_height(L1, L2, sin_theta1, cos_theta1, sin_theta2, cos_theta2):
        """
        Computes the height of the second pole in the Double Inverted Pendulum environment.
        
        Parameters:
        - L1: Length of the first pole
        - L2: Length of the second pole
        - sin_theta1: Sine of the angle between the cart and the first pole
        - cos_theta1: Cosine of the angle between the cart and the first pole
        - sin_theta2: Sine of the angle between the two poles
        - cos_theta2: Cosine of the angle between the two poles
        
        Returns:
        - y2: Height of the second pole's top end
        """
        y1 = L1 * cos_theta1
    
        # Compute height of the second pole's top using trigonometric sum identity
        y2 = y1 + L2 * (cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2)
    
        return y2 + .196 # add back the height of the cart

    
    # assumed from the docs.
    L1 = 0.5
    L2 = 0.5
    sin_theta1 = next_obs[:, 1]
    cos_theta1 = next_obs[:, 3]

    sin_theta2 = next_obs[:, 2]
    cos_theta2 = next_obs[:, 4]



    y_values = compute_second_pole_height(L1, L2, sin_theta1, cos_theta1, sin_theta2, cos_theta2)

    not_done = torch.isfinite(next_obs).all(-1) * torch.tensor(y_values > 1)
    done = ~not_done

    done = done[:, None]
    return done
    

def no_termination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    done = torch.Tensor([False]).repeat(len(next_obs)).bool().to(next_obs.device)
    done = done[:, None]
    return done



def reacher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    pass



def walker2d(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done


def ant(act: torch.Tensor, next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2

    x = next_obs[:, 0]
    not_done = torch.isfinite(next_obs).all(-1) * (x >= 0.2) * (x <= 1.0)

    done = ~not_done
    done = done[:, None]
    return done


def humanoid(act: torch.Tensor, next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2

    z = next_obs[:, 0]
    done = (z < 1.0) + (z > 2.0)

    done = done[:, None]
    return done



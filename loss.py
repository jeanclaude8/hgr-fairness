from maxcorr import indicator 
import torch

def costum_loss_1(x, x_hat, z, alpha, beta, epsilon, method='mean'):
    ind = indicator(semantics='hgr', algorithm='sk',backend='torch')
    original_correlation = []
    sensitive_correlation = []
    for i in range(len(x[1])):
        original_correlation.append(1-(ind.compute(x_hat[:, i],x[:, i])))
        sensitive_correlation.append(ind.compute(x_hat[:, i],z) - epsilon)
    original_correlation = torch.stack(original_correlation)
    sensitive_correlation = torch.stack(sensitive_correlation)
    if method == 'mean':
        constraint = torch.relu(torch.mean(sensitive_correlation))
        loss = alpha * torch.mean(original_correlation) + beta * constraint
    if method == 'max':
        constraint = torch.relu(torch.max(sensitive_correlation))
        loss = alpha * torch.max(original_correlation) + beta * constraint
    return loss, constraint


def costum_loss_2(x, x_hat, z, alpha, beta, epsilon):
    ind = indicator(semantics='hgr', algorithm='sk', backend='torch')
    original_correlation = 1-(ind.compute(x_hat,x))
    sensitive_correlation = ind.compute(x_hat,z) - epsilon
    constraint = torch.relu(sensitive_correlation)
    loss = alpha * original_correlation + beta * constraint
    return loss, constraint

def costum_loss_3(x, x_hat, z, alpha, beta, epsilon, method='mean'):
    ind = indicator(semantics='hgr', algorithm='sk', backend='torch')
    original_correlation = []
    sensitive_correlation = []
    for i in range(len(x[1])):
        original_correlation.append(epsilon-(ind.compute(x_hat[:, i],x[:, i])))
        sensitive_correlation.append(ind.compute(x_hat[:, i],z))
    original_correlation = torch.stack(original_correlation)
    sensitive_correlation = torch.stack(sensitive_correlation)
    if method == 'mean':
        constraint = torch.relu(torch.mean(original_correlation))
        loss = alpha * torch.mean(sensitive_correlation) + beta * constraint
    if method == 'max':
        constraint = torch.relu(torch.mean(original_correlation))
        loss = alpha * torch.max(sensitive_correlation) + beta * constraint
    return loss, constraint

def costum_loss_4(x, x_hat, z, alpha, beta, epsilon):
    ind = indicator(semantics='hgr', algorithm='sk', backend='torch')
    original_correlation = epsilon-(ind.compute(x_hat,x))
    sensitive_correlation = ind.compute(x_hat,z) 
    constraint = torch.relu(original_correlation)
    loss = alpha *sensitive_correlation + beta * constraint
    return loss, constraint
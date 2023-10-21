from tools import MultiItemAverageMeter
from network.processing import FeatureMapIntegrating, FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights,\
    FeatureMapLocalizedIntegratingNoRelu

def train(base, loaders, config):

    base.set_train()
    loader = loaders.loader
    meter = MultiItemAverageMeter()
    for i, data in enumerate(loader):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), \
                           cids.to(base.device).long()
        if config.module == 'B':
            features_map = base.model(imgs)
            bn_features, cls_score = base.classifier(features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)

            total_loss = ide_loss

            base.model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()
            base.classifier_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
            })

        elif config.module == 'CIP_w_Q_L':

            features_map = base.model(imgs)
            integrating_features_map, integrating_pids = FeatureMapIntegrating(config).__call__(features_map, pids)
            bn_features, cls_score = base.classifier(features_map)
            integrating_bn_features, integrating_cls_score = base.classifier2(integrating_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)
            integrating_ide_loss = base.pid_creiteron(integrating_cls_score, integrating_pids)
            integrating_reasoning_loss = base.reasoning_creiteron(bn_features, integrating_bn_features)

            total_loss = ide_loss + integrating_ide_loss + \
                         config.lambda1 * integrating_reasoning_loss

            base.model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            base.classifier2_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()
            base.classifier_optimizer.step()
            base.classifier2_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'integrating_pid_loss': integrating_ide_loss.data,
                          'integrating_reasoning_loss': integrating_reasoning_loss.data,
                          })

        elif config.module == 'CIP_w_L':

            features_map = base.model(imgs)
            bn_features, cls_score = base.classifier(features_map)
            quantified_features_map, quantified_integrating_features_map, integrating_pids = \
                FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights(config).__call__(features_map, cls_score, pids)

            quantified_integrating_bn_features, quantified_integrating_cls_score = \
                base.classifier2(quantified_integrating_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)
            quantified_integrating_ide_loss = base.pid_creiteron(quantified_integrating_cls_score, integrating_pids)
            quantified_integrating_reasoning_loss = base.reasoning_creiteron(bn_features,
                                                                             quantified_integrating_bn_features)

            total_loss = ide_loss + quantified_integrating_ide_loss + \
                         config.lambda1 * quantified_integrating_reasoning_loss

            base.model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            base.classifier2_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()
            base.classifier_optimizer.step()
            base.classifier2_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'quantified_integrating_pid_loss': quantified_integrating_ide_loss.data,
                          'quantified_integrating_reasoning_loss': quantified_integrating_reasoning_loss.data,
                          })

        elif config.module == 'CIP_w_Q':

            features_map = base.model(imgs)
            bn_features, cls_score = base.classifier(features_map)
            localized_features_map, localized_integrating_features_map, integrating_pids = \
                FeatureMapLocalizedIntegratingNoRelu(config).__call__(features_map, pids, base)

            localized_integrating_bn_features, localized_integrating_cls_score = \
                base.classifier2(localized_integrating_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)
            localized_integrating_ide_loss = base.pid_creiteron(localized_integrating_cls_score, integrating_pids)
            localized_integrating_reasoning_loss = base.reasoning_creiteron(bn_features,
                                                                            localized_integrating_bn_features)

            total_loss = ide_loss + localized_integrating_ide_loss + \
                         config.lambda1 * localized_integrating_reasoning_loss

            base.model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            base.classifier2_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()
            base.classifier_optimizer.step()
            base.classifier2_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'localized_integrating_pid_loss': localized_integrating_ide_loss.data,
                          'localized_integrating_reasoning_loss': localized_integrating_reasoning_loss.data,
                          })

        elif config.module == 'CIP':

            features_map = base.model(imgs)
            bn_features, cls_score = base.classifier(features_map)

            localized_features_map, localized_integrating_features_map, integrating_pids = \
                FeatureMapLocalizedIntegratingNoRelu(config).__call__(features_map, pids, base)

            localized_bn_features, localized_cls_score = base.classifier(localized_features_map)

            quantified_localized_features_map, quantified_localized_integrating_features_map, _ = \
                FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights(config).__call__(localized_features_map,
                                                                                      localized_cls_score, pids)
            quantified_localized_integrating_bn_features, quantified_localized_integrating_cls_score = \
                base.classifier2(quantified_localized_integrating_features_map)

            ide_loss = base.pid_creiteron(cls_score, pids)
            quantified_localized_integrating_ide_loss = base.pid_creiteron(
                quantified_localized_integrating_cls_score, integrating_pids)
            quantified_localized_integrating_reasoning_loss = base.reasoning_creiteron(bn_features,
                                                                    quantified_localized_integrating_bn_features)

            total_loss = ide_loss + quantified_localized_integrating_ide_loss + \
                         config.lambda1 * quantified_localized_integrating_reasoning_loss

            base.model_optimizer.zero_grad()
            base.classifier_optimizer.zero_grad()
            base.classifier2_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()
            base.classifier_optimizer.step()
            base.classifier2_optimizer.step()

            meter.update({'pid_loss': ide_loss.data,
                          'quantified_localized_integrating_pid_loss': quantified_localized_integrating_ide_loss.data,
                          'quantified_localized_integrating_reasoning_loss': quantified_localized_integrating_reasoning_loss.data,
                          })

    return meter.get_val(), meter.get_str()








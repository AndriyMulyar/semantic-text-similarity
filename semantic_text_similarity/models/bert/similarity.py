from pytorch_transformers import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel

import torch
from torch import nn

from scipy.stats import pearsonr

from . import bert_sentence_pair_preprocessing

import logging, os

class BertSimilarityRegressor(BertPreTrainedModel):
    def __init__(self, bert_model_config: BertConfig):
        super(BertSimilarityRegressor, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        linear_size = bert_model_config.hidden_size
        if bert_model_config.pretrained_config_archive_map['additional_features'] is not None:
            linear_size+=bert_model_config.pretrained_config_archive_map['additional_features']

        self.regression = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(linear_size, 1)
        )

        self.apply(self.init_weights)


    def forward(self, input_ids, token_type_ids, attention_masks, additional_features=None):
        """
        Feed forward network with one hidden layer.
        :param input_ids:
        :param token_type_ids:
        :param attention_masks:
        :return:
        """
        _,pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks)

        if additional_features is not None:
            pooled_output = torch.cat((pooled_output, additional_features),dim=1)

        return self.regression(pooled_output)


class BertSimilarity():
    """
    A class implementing the training and evaluation of a fine tuned BERT model for sentence pair similarity.
    """

    def __init__(self, args= None, device='cuda', bert_model_path='bert-base-uncased', batch_size=10, learning_rate = 5e-5, weight_decay=0, additional_features=None):
        if args is not None:
            self.args = vars(args)

        assert device in ['cuda', 'cpu']

        if not args:
            self.args = {}
            self.args['bert_model_path'] = bert_model_path
            self.args['device'] = device
            self.args['learning_rate'] = learning_rate
            self.args['weight_decay'] = weight_decay
            self.args['batch_size'] = batch_size

        self.log = logging.getLogger()



        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_model_path'])
        if os.path.exists(self.args['bert_model_path']):
            if os.path.exists(os.path.join(self.args['bert_model_path'], CONFIG_NAME)):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], CONFIG_NAME))
            elif os.path.exists(os.path.join(self.args['bert_model_path'], 'bert_config.json')):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], 'bert_config.json'))
            else:
                raise ValueError("Cannot find a configuration for the BERT model you are attempting to load.")

        self.loss_function = torch.nn.MSELoss()

        config.pretrained_config_archive_map['additional_features'] = additional_features

        self.regressor_net = BertSimilarityRegressor.from_pretrained(self.args['bert_model_path'], config=config)
        self.optimizer = torch.optim.Adam(
            self.regressor_net.parameters(),
            weight_decay=self.args['weight_decay'],
            lr=self.args['learning_rate']
        )
        self.log.info('Initialized BertSentencePairSimilarity model from %s' % self.args['bert_model_path'])



    def predict(self, data: list, additional_features=None, return_predictions=True):
        """
        Given a list of sentence pair instances, makes predictions
        :param data:
        :param return predictions or log to model directory
        :return:
        """

        self.regressor_net.to(device=self.args['device'])
        self.regressor_net.eval()

        with torch.no_grad():

            if data is not None and isinstance(data, list):
                if isinstance(data[0], dict):
                    input_ids_eval, token_type_ids_eval, attention_masks_eval, correct_scores_eval = bert_sentence_pair_preprocessing(
                        data, self.bert_tokenizer)
                elif isinstance(data[0], tuple):
                    input_ids_eval, token_type_ids_eval, attention_masks_eval, correct_scores_eval = bert_sentence_pair_preprocessing(
                        [{'sentence_1': s1, 'sentence_2':s2} for s1, s2 in data], self.bert_tokenizer)
                else:
                    raise ValueError("Data must be a list of sentence pair tuples")

            predictions = torch.empty_like(correct_scores_eval)
            for i in range(0, input_ids_eval.shape[0], self.args['batch_size']):
                input_id_eval = input_ids_eval[i:i + self.args['batch_size']].to(device=self.args['device'])
                token_type_id_eval = token_type_ids_eval[i:i + self.args['batch_size']].to(device=self.args['device'])
                attention_mask_eval = attention_masks_eval[i:i + self.args['batch_size']].to(device=self.args['device'])

                if additional_features is not None:
                    additional_feature = additional_features[i:i + self.args['batch_size']].to(device=self.args['device'])
                else:
                    additional_feature = None

                predicted_score = self.regressor_net(input_id_eval, token_type_id_eval, attention_mask_eval, additional_features=additional_feature)
                predictions[i:i + self.args['batch_size']] = predicted_score


        if all(torch.isnan(correct_scores_eval)) or return_predictions: #no correct score labels are present, return the predictions
            return predictions.cpu().view(-1).numpy()
        else:
            self.log.info('Evaluating on Epoch %i' % (self.epoch))
            scores = {'pearson': 0}
            scores['pearson'] = \
                pearsonr(predictions.cpu().view(-1).numpy(), correct_scores_eval.view(-1).numpy())[0]

            with open(os.path.join(self.args['model_directory'], "eval_%s.csv" % self.epoch), 'w') as eval_results:
                eval_results.write("sentence_1\tsentence_2\tannotator_score\tpredicated_score\n")

                for idx, row in enumerate(torch.cat((correct_scores_eval, predictions.cpu()), dim=1)):
                    eval_results.write(
                        "%s\t%s\t%f\t%f\n" % (data[idx]['sentence_1'], data[idx]['sentence_2'], row[0].item(), row[1].item()))
                eval_results.write("Pearson\t%f" % (scores['pearson']))

            self.log.info("Pearson: %f" % (scores['pearson']))
        self.regressor_net.train()
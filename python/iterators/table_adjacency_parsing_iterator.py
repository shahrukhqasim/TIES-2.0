from iterators.iterator_interface import Iterator
from overrides import overrides
import tensorflow as tf
from libs.configuration_manager import ConfigurationManager as gconfig
from models.model_interface import ModelInterface
import subprocess
import os


class TableAdjacencyParsingIterator (Iterator):
    def __init__(self):
        self.from_scratch = gconfig.get_config_param("from_scratch", type="bool")
        self.summary_path = gconfig.get_config_param("summary_path", type="str")
        self.model_path = gconfig.get_config_param("model_path", type="str")
        self.train_for_iterations = gconfig.get_config_param("train_for_iterations", type="int")
        self.validate_after = gconfig.get_config_param("validate_after", type="int")
        self.save_after_iterations = gconfig.get_config_param("save_after_iterations", type="int")
        self.test_out_path = gconfig.get_config_param("test_out_path", type="str")


    def clean_summary_dir(self):
        print("Cleaning summary dir")
        for the_file in os.listdir(self.summary_path):
            file_path = os.path.join(self.summary_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    @overrides
    def train(self):
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

        model  = ModelInterface()
        saver = model.get_saver()

        if self.from_scratch:
            subprocess.call("mkdir -p %s"%(self.summary_path), shell=True)
            subprocess.call("mkdir -p %s"%(self.test_out_path), shell=True)
            subprocess.call("mkdir -p %s"%(os.path.join(self.test_out_path, 'ops')), shell=True)
        else:
            self.clean_summary_dir()

        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

            if not self.from_scratch:
                saver.restore(sess, self.model_path)
                print("\n\nINFO: Loading model\n\n")
                with open(self.model_path + '.txt', 'r') as f:
                    iteration_number = int(f.read())
            else:
                iteration_number = 0

            print("Starting iterations")
            while iteration_number < self.train_for_iterations:
                feed_dict = model.get_feed_dict(iteration_number)
                fetches = model.get_fetches(iteration_number)

                result = sess.run(fetches, feed_dict=feed_dict)
                eval_summary = model.post_fetch(result, iteration_number=iteration_number)

                if iteration_number % self.validate_after == 0:
                    feed_dict = model.get_feed_dict_validation(iteration_number)
                    fetches = model.get_fetches_validation(iteration_number)

                    result = sess.run(fetches, feed_dict=feed_dict)
                    eval_summary_validation = model.post_fetch_validation(result, iteration_number=iteration_number)
                    summary_writer.add_summary(eval_summary_validation, iteration_number)

                iteration_number += 1
                summary_writer.add_summary(eval_summary, iteration_number)
                if iteration_number % self.save_after_iterations == 0:
                    print("\n\nINFO: Saving model\n\n")
                    saver.save(sess, self.model_path)
                    with open(self.model_path + '.txt', 'w') as f:
                        f.write(str(iteration_number))



            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)


        return super(TableAdjacencyParsingIterator, self).train()

    @overrides
    def test(self):
        return super(TableAdjacencyParsingIterator, self).test()

    @overrides
    def profile(self):
        return super(TableAdjacencyParsingIterator, self).profile()

    @overrides
    def visualize(self):
        return super(TableAdjacencyParsingIterator, self).visualize()


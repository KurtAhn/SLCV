import tensorflow as tf
from os import path


class Model:
    def __init__(self, name, **kwargs):
        self._name = name
        if 'mdldir' in kwargs:
            self._restore(kwargs['mdldir'], kwargs.get('epoch', 0))
        else:
            self._create(**kwargs)

    def _restore(self, mdldir, epoch):
        g = tf.get_default_graph()
        meta = tf.train.import_meta_graph(path.join(mdldir, '_.meta'),
                                          clear_devices=True)
        if epoch:
            with open(path.join(mdldir, 'checkpoint'), 'w') as f:
                f.write('model_checkpoint_path: "' +
                        path.join(mdldir, '_-{}'.format(epoch)) + '"')
        meta.restore(tf.get_default_session(),
                     tf.train.latest_checkpoint(mdldir))

    def save(self, saver, mdldir, epoch):
        if epoch == 1:
            tf.train.export_meta_graph(
                filename=path.join(mdldir, '_.meta')
            )
        saver.save(tf.get_default_session(),
                   path.join(mdldir, '_'),
                   global_step=epoch,
                   write_meta_graph=False)

    @property
    def name(self):
        return self._name

    def __getitem__(self, k):
        g = tf.get_default_graph()
        try:
            return g.get_tensor_by_name('{}/{}:0'.format(self.name, k))
        except KeyError:
            try:
                return g.get_operation_by_name('{}/{}'.format(self.name, k))
            except KeyError:
                raise KeyError('Nonexistent name: {}/{}'.format(self.name, k))

    def __getattr__(self, a):
        return self[a]

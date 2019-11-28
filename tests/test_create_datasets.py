from datasets.datasets_base import DatasetsBase

def test_create_datasetsbase():
    datasetbase = DatasetsBase("/path/to/datasetfile.txt")
    datasetbase.setup()
# test_create_datasetsbase()


def test_create_dataset():
    class ChildClass(DatasetsBase):
        def setup(self):
            self.data = ["/path/to/the/image1.jpg","/path/to/the/image2.jpg"]
            self.labels = [[11,22,33,44],[55,66,77,88]]
            super(ChildClass,self).setup()

    dataset = ChildClass("/path/to/trainset.txt")
    dataset.setup()

    print(dataset[0])
test_create_dataset()
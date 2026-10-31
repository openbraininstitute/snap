import bluepysnap._doctools as test_module


class TestClass:
    __test__ = False

    def __init__(self):
        """TestClass"""
        self.foo_val = 0
        self.bar_val = 1

    def foo(self):
        """foo function for TestClass"""
        return self.foo_val

    def bar(self):
        """bar function for TestClass

        Returns:
            TestClass: return a TestClass
        """
        return self.bar_val

    def foo_bar(self):
        pass


class TestClassA(
    TestClass,
    metaclass=test_module.DocSubstitutionMeta,
    source_word="TestClass",
    target_word="TestClassA",
):
    """New class with changed docstrings."""


class TestClassB(
    TestClass,
    metaclass=test_module.DocSubstitutionMeta,
    source_word="TestClass",
    target_word="TestClassB",
):
    """New class with changed docstrings."""


class TestClassC(TestClassA):
    def bar(self):
        """Is overrode correctly"""
        return 42


def test_DocSubstitutionMeta():
    default = TestClass()
    foo_doc = default.foo.__doc__
    bar_doc = default.bar.__doc__
    assert foo_doc is not None
    assert bar_doc is not None
    tested = TestClassA()
    assert tested.foo.__doc__ == foo_doc.replace("TestClass", "TestClassA")
    expected = bar_doc.replace("TestClass", "TestClassA")
    assert tested.bar.__doc__ == expected
    assert tested.foo_bar.__doc__ is None
    assert tested.__dict__ == default.__dict__

    # do not override the mother class docstrings
    tested = TestClassB()
    assert tested.foo.__doc__ == foo_doc.replace("TestClass", "TestClassB")
    expected = bar_doc.replace("TestClass", "TestClassB")
    assert tested.bar.__doc__ == expected
    assert tested.foo_bar.__doc__ is None
    assert tested.__dict__ == default.__dict__

    # I can inherit from a class above
    tested = TestClassC()
    assert tested.foo.__doc__ == TestClassA.foo.__doc__
    assert tested.bar.__doc__ == "Is overrode correctly"
    assert tested.bar() == 42

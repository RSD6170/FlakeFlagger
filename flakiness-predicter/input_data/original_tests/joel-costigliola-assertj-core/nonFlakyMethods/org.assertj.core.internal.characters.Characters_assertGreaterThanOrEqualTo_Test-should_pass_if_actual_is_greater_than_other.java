@Test public void should_pass_if_actual_is_greater_than_other(){
  characters.assertGreaterThanOrEqualTo(someInfo(),'b','a');
}
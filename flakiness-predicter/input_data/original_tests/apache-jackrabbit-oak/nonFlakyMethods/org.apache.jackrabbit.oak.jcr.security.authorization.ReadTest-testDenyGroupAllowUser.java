@Test public void testDenyGroupAllowUser() throws Exception {
  deny(path,getTestGroup().getPrincipal(),readPrivileges);
  allow(path,testUser.getPrincipal(),readPrivileges);
  assertTrue(testSession.nodeExists(path));
}